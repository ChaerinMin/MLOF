import argparse
import os
import random
import socket
import time

from core.utils.utils import str2bool
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import learn2learn as l2l
import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils import data

import core.datasets as datasets
import evaluate
# from torch import optim
import wandb
from core.advanced_loss.flow_loss import unFlowLoss
from core.losses import PhotoSmoothLoss
from core.network import RAFTGMA
from core.utils.logger import flow_img
# from core.utils.logger import Logger, flow_img
from pretrain import count_parameters, fetch_optimizer, sequence_loss

# from core.utils.utils import InputPadder
# from validate import theta_prime, validate_maml
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5"


print(socket.gethostname())

def main(args):
    root = './datasets'

    chkp = torch.load(args.maml_pth)
    model = torch.nn.DataParallel(RAFTGMA(args), device_ids=args.gpus) 
    # model = RAFTGMA(args)
    if args.fofrom_theta0:
        tmp_ch = torch.load(args.maml_pth)
        model.module.load_state_dict(tmp_ch)
    else:
        try:
            model.load_state_dict(chkp['model_state_dict'])
            model = l2l.algorithms.MAML(model,lr=args.lr,allow_nograd=True,first_order=args.first_order) 
        except:
            model = l2l.algorithms.MAML(model,lr=args.lr,allow_nograd=True,first_order=args.first_order) 
            model.load_state_dict(chkp['model_state_dict'])

    model.cuda()
    model.eval()
    wandb.init(config=args,name=str(args.name)+"_id"+str(args.one_image),group='maml',dir='/tmp')
    wandb.watch(model)
    if args.name == 'debug':
        os.environ['WANDB_MODE'] = 'offline'
        
    aug_params={'crop_size': args.image_size, 'min_scale': 0, 'max_scale': 0, 'do_flip': False}

    results = []
    initial_results=[]
    num_steps = args.num_steps
    assert args.dataset == args.validation

    if args.dataset == 'kitti':
        root = os.path.join(root,'KITTI')
        ft_dataset = datasets.SingleKITTI(img_id=args.one_image,root=root,aug_params = aug_params,split='training') 
    elif args.dataset == 'sintel':
        root = os.path.join(root,'Sintel')
        ft_dataset = datasets.SintelSelect([args.one_image],aug_params=aug_params,root=root,split='training',dstype=args.dstype)
    else:
        raise NotImplementedError

    ft_loader = data.DataLoader(ft_dataset, batch_size=args.batch_size,
                                pin_memory=True, shuffle=True, num_workers=8, drop_last=True)
    # model_adapt = model.clone()
    model_adapt = torch.nn.DataParallel(RAFTGMA(args), device_ids=args.gpus)
    try:
        model_adapt.load_state_dict(chkp['model_state_dict'])
        model_adapt = l2l.algorithms.MAML(model_adapt,lr=args.lr,allow_nograd=True,first_order=args.first_order)
    except:
        model_adapt = l2l.algorithms.MAML(model_adapt,lr=args.lr,allow_nograd=True,first_order=args.first_order)
        model_adapt.load_state_dict(chkp['model_state_dict'])        

    model_adapt.train()
    optimizer, scheduler = fetch_optimizer(args, model_adapt)
    scaler = GradScaler(enabled=args.mixed_precision)
    # loss_fun = PhotoSmoothLoss(ptmodel=model,args=args)
    if args.loss_fn == 'photometric':
        loss_fun = PhotoSmoothLoss(ptmodel=model,args=args) 
    elif args.loss_fn== 'arloss':
        loss_fun = unFlowLoss(args,model)
    else:
        loss_fun = None
    valcount=0
    initial_result = evaluate.uni_validate(args,model_adapt.module,count=valcount,iters=12,root=root)
    initial_results.append(initial_result['kitti_epe'])
    ft_epe = []
    ft_loss = []
    val_epe=[]
    for iter in range(num_steps):  
        for i_batch, data_blob in enumerate(ft_loader):
            # data_blob = ft_dataset[0]
            tic = time.time()
            image1, image2, flow, valid = [x.cuda() for x in data_blob[:-1]]

            optimizer.zero_grad()

            flow_pred = model_adapt(image1, image2) 

            if args.loss_fn == 'photometric':
                loss,metrics = loss_fun(image1, image2, flow_pred,flow,valid,args.gamma,model_adapt) #image1 (4,3,384,512)
            elif args.loss_fn == 'L1':
                loss, metrics = sequence_loss(flow_pred, flow, valid, args.gamma)
            elif args.loss_fn == 'arloss':
                flow_back = model_adapt(image2,image1)
                fb_flow = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flow_pred, flow_back)]
                image_pair = torch.cat([image1, image2], 1) #(B,6,H,W)
                loss,metrics, rangemap2 = loss_fun(fb_flow,image_pair,flow,valid,model_adapt)

            scaler.scale(loss).backward() 
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model_adapt.parameters(), args.clip)
            scaler.step(optimizer) 
            scheduler.step()
            scaler.update()
            toc = time.time()

            metrics['ztime'] = toc - tic
            valcount += 1
            val_result = evaluate.uni_validate(args,model_adapt.module,count=valcount,iters=12,root=root) 
            val_epe.append(val_result['kitti_epe'])
            ft_epe.append(metrics['epe'])
            ft_loss.append(metrics['loss_total'])
            if iter %10 == 9 or iter in np.arange(args.little_step):
                ft_epe = []
                ft_loss = []
                val_epe=[]
        print(f"image id {args.one_image} : {iter}/{num_steps}")
    model_adapt.eval()
    result_dict = evaluate.uni_validate(args,model_adapt.module,count=valcount,iters=12,root=root) 
    results.append(result_dict['kitti_epe'])



if __name__ == '__main__':
    timename = time.strftime('%m%d_%H%M%S', time.localtime())
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=False, help='use mixed precision')
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--loss_iters', default=False, help = 'whether to compute losses from all iters during fine-tuning') 

    parser.add_argument('--dstype', default='final',help="determines which dataset to use for training")


    #Optimization
    parser.add_argument('--batch_size', type=int, default=1) #4
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--val_freq', type=int, default=100,
                        help='validation frequency') 
    parser.add_argument('--print_freq', type=int, default=20,
                        help='printing frequency') 

    #arloss
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

    #fine-tuning
    parser.add_argument('--wkeeper',type=float, default=10, help='factor of norm(f0-fi) for fine-tuning')
    parser.add_argument('--loss_fn', default='arloss', help = 'photometric,arloss or L1') 
    parser.add_argument('--little_step',default=20,type=int)
    parser.add_argument('--optimizer', type=str,default='adam', help="name your experiment")
    parser.add_argument('--occ_aware',default=False,help="if True, occlusion aware loss")
    parser.add_argument('--meta_algo',default='maml',help='maml, metasgd')
    parser.add_argument('--fofrom_theta0',default=False)
    parser.add_argument('--convgru',default='',help="if count, aux losses for all gru iterations. else '' ")
    parser.add_argument('--model', help="restore checkpoint", default=None)#checkpoints/gma-things.pth 
    parser.add_argument('--clip_flow',type=float, default=7, help='flow magnitude clamping for visualization')

    parser.add_argument('--dataset', default='sintel',help="dataset for fine-tuning")
    parser.add_argument('--validation', type=str, default='sintel')
    parser.add_argument('--maml_pth', default="sintel_ml")
    parser.add_argument('--output', type=str, default='./results/', help='output directory to save checkpoints and plots')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--num_steps', type=int, default=3) 
    parser.add_argument('--image_size', type=int, nargs='+', default=[368, 768], help='image size will be automatically set in the code.') 
    parser.add_argument('--one_image',type=int, default=0, help='the index of a single image to do theta_zero^star, theta^{ft,star}, or theta^{ml,star}')
    parser.add_argument('--name',help="name your experiment", default="sintel_ml_star")
    parser.add_argument('--load_step',default=50,type=int)
    parser.add_argument('--crossval',default=1,type=int,help="1,2,3")
    parser.add_argument('--fifo',default=True,help='|fi-fo| term',type=str2bool)
    parser.add_argument('--first_order',default=True,type=str2bool)
    parser.add_argument('--force_size',default=False,type=str2bool)

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    args.save_pth = os.path.join(args.output,args.maml_pth,"ml_star"+str(args.load_step))
    args.maml_pth = os.path.join(args.output,args.maml_pth,f'step{args.load_step}.pth')
    if not os.path.exists(args.save_pth):
        os.mkdir(args.save_pth)
    args.save_pth = os.path.join(args.save_pth,str(args.one_image))
    if not os.path.exists(args.save_pth):
        os.mkdir(args.save_pth)

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
    main(args) 