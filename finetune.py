import argparse
import os
import socket
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time

import learn2learn as l2l
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils import data

import core.datasets as datasets
import evaluate
import wandb
from core.advanced_loss.flow_loss import unFlowLoss
from core.losses import PhotoSmoothLoss
from core.network import RAFTGMA
from core.utils.logger import Logger, error_map, flow_img
from core.utils.utils import remove_dir, str2bool
from exampler.loss import Exampler_Loss
from exampler.simple import *
from pretrain import count_parameters, fetch_optimizer, sequence_loss

print(socket.gethostname())



def validate(model, args,count, logger,root):
    model.eval()
    results = {}
    
    if args.validation == 'chairs':
        results.update(evaluate.validate_chairs(model.module, args.iters)) 
    elif args.validation == 'sintel': 
        if args.one_image>-1:
            results.update(evaluate.uni_validate(args,model.module,count,args.iters,root=root))
        else:
            results.update(evaluate.validate_sintel(model.module, args.iters))
    elif args.validation == 'kitti':
        if args.one_image>-1:
            results.update(evaluate.validate_single_kitti(model.module, count,logger, args.iters,length=1,args=args,root=root)) 
        else:    
            results.update(evaluate.validate_kitti(model.module, logger, args.iters,length=10,args=args))
    

    for key in results.keys(): 
        if key not in logger.val_results_dict.keys():
            logger.val_results_dict[key] = []
        logger.val_results_dict[key].append(results[key])

    model.train()
    return results


def fine_tune(model, ft_loader, optimizer, scheduler, logger, scaler,loss_fun, args,root,valcount=-1):
    images=[]
    gts=[]
    rgb =[]
    errors=[]
    errors_clip=[]
    range_occ=[]
    range_noc=[]
    if args.exampler_id:
        ex_loss = Exampler_Loss(args.exampler_id,args)
    for i_batch, data_blob in enumerate(ft_loader):
        tic = time.time()
        
        image1, image2, flow, valid = [x.cuda() for x in data_blob[:-1]]
        gt_occ=None

        optimizer.zero_grad()

        flow_pred = model(image1, image2) 
        if args.loss_fn == 'photometric':
            loss,metrics = loss_fun(image1, image2, flow_pred,flow,valid,args.gamma,model) #image1 (4,3,384,512)
        elif args.loss_fn == 'L1':
            loss, metrics = sequence_loss(flow_pred, flow, valid, args.gamma)
        elif args.loss_fn == 'arloss':
            flow_back = model(image2,image1)
            fb_flow = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flow_pred, flow_back)]
            image_pair = torch.cat([image1, image2], 1) #(B,6,H,W)
            loss,metrics,rangemap2 = loss_fun(fb_flow,image_pair,flow,valid,model,gt_occ) 
        
        if args.exampler_id:
            ex_loss.compute_pi(args.one_image)
            exampler_loss = ex_loss(flow_pred)
            loss += exampler_loss 

        scaler.scale(loss).backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
  
        scaler.step(optimizer) 
        scheduler.step()
        scaler.update()
        toc = time.time()
        valcount += 1

        # metrics['ztime'] = toc - tic
        rgb.append(wandb.Image(image1))
        images.append(wandb.Image(flow_img(flow_pred[-1])))
        gts.append(wandb.Image(flow_img(flow)))
        er, er_clip = error_map(flow_pred[-1],flow)
        errors.append(wandb.Image(er))
        errors_clip.append(wandb.Image(er_clip))

        range_occ = None
        range_noc = None
        logger.push(metrics,images,gts,rgb,errors,errors_clip,range_occ,range_noc) 

        validate(model, args, valcount,logger,root=root) 

    return valcount

def finetune_common(args, model,fixed_model):

    if args.dataset == 'sintel': 
        root = os.path.join(root,'Sintel')
        if args.one_image>-1:
            aug_params = {'crop_size': args.image_size, 'min_scale': 0, 'max_scale': 0, 'do_flip': False}
            ft_dataset = datasets.SintelSelect([args.one_image],aug_params = aug_params,dstype=args.dstype,root=root)
        else:
            raise NotImplementedError('')
    elif args.dataset == 'chairs':
        ft_dataset = datasets.FlyingChairs(split='validation')
    elif args.dataset == 'kitti':
        root = os.path.join(root,'KITTI')
        if args.one_image>-1:
            ft_dataset = datasets.SingleKITTI(args.one_image,aug_params={'crop_size':args.image_size,'min_scale':0,'max_scale':0,'do_flip':False},root=root)
        else:
            ft_dataset = datasets.KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True},root=root) 
    ft_loader = data.DataLoader(ft_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=True, num_workers=8, drop_last=True)
    optimizer, scheduler = fetch_optimizer(args, model)
    scaler = GradScaler(enabled=args.mixed_precision) 
    logger = Logger(model, scheduler, args)
    if args.loss_fn == 'photometric':
        loss_fun = PhotoSmoothLoss(ptmodel=fixed_model,args=args)
    elif args.loss_fn== 'arloss':
        loss_fun = unFlowLoss(args,fixed_model)
    else:
        loss_fun = None
    
    valcount=0
    logger.if_write('initial') 
    pt_result = validate(model, args, valcount,logger,root=root) 
    logger.if_write('')


    while logger.total_steps <= args.num_steps:
        valcount=fine_tune(model, ft_loader, optimizer, scheduler, logger, scaler,loss_fun, args,root, valcount)
        print("epochs {} done".format(logger.total_steps))

    logger.if_write('final')
    ft_result = validate(model,args,valcount,logger,root=root)
    logger.if_write('')


    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    timename = time.strftime('%m%d_%H%M%S', time.localtime())
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=False, help='use mixed precision')
    parser.add_argument('--model_name')
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--alpha', type=float, default=10, help='for arloss')
    parser.add_argument('--w_l1', type=float, default=0.15, help='for arloss')
    parser.add_argument('--w_ssim', type=float, default=0.85, help='for arloss')
    parser.add_argument('--w_ternary', type=float, default=0.0, help='for arloss')
    parser.add_argument('--w_smooth', type=float, default=75.0, help='for arloss')
    parser.add_argument('--smooth_2nd', default=True, help = 'for arloss')
    parser.add_argument('--with_bk', default=True, help = 'for arloss')
    parser.add_argument('--occ_from_back', default=True, help = 'for arloss')
    parser.add_argument('--warp_pad', type=str,default='border', help = 'for arloss')
    parser.add_argument('--w_scales', type=float,default=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],nargs='+', help = 'for arloss')
    parser.add_argument('--w_sm_scales', type=float,default=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],nargs='+', help = 'for arloss')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--loss_iters', default=False, help = 'whether to compute losses from all iters during fine-tuning')

    parser.add_argument('--dstype', default='final',help="determines which dataset to use for training")

    parser.add_argument('--batch_size', type=int, default=1) #4
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--save_freq', type=int, default=2,
                        help='checkpoint frequency') # 100
    parser.add_argument('--print_freq', type=int, default=1,
                        help='logging and image-saving frequency') # 100

    parser.add_argument('--wkeeper',type=float, default=10, help='factor of norm(f0-fi) for fine-tuning')
    parser.add_argument('--loss_fn', default='arloss', help = 'photometric,arloss or L1') 
    parser.add_argument('--optimizer', type=str,default='adam', help="name your experiment")
    parser.add_argument('--occ_aware',default=False,help="if True, occlusion aware loss")
    parser.add_argument('--simple_exemplar',default=False,help="the simplest experiment for the exemplar idea")
    # parser.add_argument('--compare_err', type=str, nargs='+', default=['','']) 

    parser.add_argument('--exampler_id',type=int,default=None)
    parser.add_argument('--meta_algo',default='maml',help='maml, metasgd')
    parser.add_argument('--convgru',default='',help="if count, aux losses for all gru iterations. else '' ")
    parser.add_argument('--fo_train',default=False,help='set fo to be trainable')
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--clip_flow',type=float, default=7, help='flow magnitude clamping for visualization')
    parser.add_argument('--image_size', type=int, nargs='+', default=[368, 768]) 
    parser.add_argument('--output', type=str, default='./results/naive_ft/', help='output directory to save checkpoints and plots')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--fifo',default=True,help='|fi-fo| term',type=str2bool)
    parser.add_argument('--crossval',default=None,type=int,help="1,2,3")
    parser.add_argument('--first_order',default=True,type=str2bool)
    parser.add_argument('--inner_steps',type=int,default=3)
    parser.add_argument('--force_size',default=False,help='avoid automatic image_size for datasets',type=str2bool)


    parser.add_argument('--dataset', help="dataset for fine-tuning",default = 'sintel')
    parser.add_argument('--validation', type=str,default='sintel')
    parser.add_argument('--num_steps', type=int, default=50) #120000
    parser.add_argument('--model', help="restore checkpoint", default = './checkpoints/gma-things.pth')
    parser.add_argument('--one_image',type=int,default=198,help='the index of a single image to do theta_zero^star, theta^{ft,star}, or theta^{ml,star}')
    parser.add_argument('--name', default='debug', help="name your experiment")


    args = parser.parse_args()

    remove_dir(args.output)
    if args.one_image >-1:
        # if args.compare_err[0]:
        #     args.output = args.output+'diff_{}'.format(args.name)
        # else:
        args.output = args.output+'{}/id{}'.format(args.name,str(args.one_image))
    else:
        args.output = args.output+'/{}_{}'.format(timename,args.name)
        
    if args.name == 'debug':
        os.environ['WANDB_MODE'] = 'offline'
    
    if args.dataset == 'kitti':
        args.image_size = [288,960]
        args.wdecay = 0.00001
        args.gamma = 0.85
    elif args.dataset == 'sintel':
        if not args.force_size:
            args.image_size = [368,768]
        args.wdecay = 0.00001
        args.gamma = 0.85
    else:
        raise NotImplementedError

    torch.manual_seed(1234)
    np.random.seed(1234)
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    args.save_pth = args.output

    print('Current cuda device:', torch.cuda.current_device())

    model = torch.nn.DataParallel(RAFTGMA(args))
    fixed_model = torch.nn.DataParallel(RAFTGMA(args))


    # if args.compare_err[0]:
    #     raise NotImplementedError
    # else:
    try:
        model.load_state_dict(torch.load(args.model))
        fixed_model.load_state_dict(torch.load(args.model))
    except:
        try:
            model.load_state_dict(torch.load(args.model)['model_state_dict'])
            fixed_model.load_state_dict(torch.load(args.model)['model_state_dict'])
        except:
            model = l2l.algorithms.MAML(model,lr=args.lr,allow_nograd=True,first_order=args.first_order) 
            fixed_model = l2l.algorithms.MAML(fixed_model,lr=args.lr,allow_nograd=True,first_order=args.first_order) 
            try:
                model.load_state_dict(torch.load(args.model)['model_state_dict'])
                fixed_model.load_state_dict(torch.load(args.model)['model_state_dict'])
            except:
                model.load_state_dict(torch.load(args.model))
                fixed_model.load_state_dict(torch.load(args.model))

            model.cuda()
            model.train()
            fixed_model.cuda()
            if not args.fo_train:
                fixed_model.eval()
    
    model.cuda()
    model.train()
    fixed_model.cuda()

    finetune_common(args, model,fixed_model) 
    print("fine-tuning has been successfully done")


