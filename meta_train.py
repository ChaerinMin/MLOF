import argparse
import copy
import os
import socket
import time

import learn2learn as l2l
import numpy as np
import torch
from PIL import Image
from torch import optim

import core.datasets as datasets
import wandb
from core.advanced_loss.flow_loss import unFlowLoss
from core.losses import PhotoSmoothLoss
from core.network import RAFTGMA
from core.utils.logger import flow_img
from core.utils.utils import remove_dir, str2bool
from pretrain import count_parameters, sequence_loss
from validate import validate_maml

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

print(socket.gethostname())


EPOCHS = 10000

def set_model(args):
    model = torch.nn.DataParallel(RAFTGMA(args), device_ids=args.gpus) 
    fixed_model = torch.nn.DataParallel(RAFTGMA(args), device_ids=args.gpus)


    later = False
    try:
        model.load_state_dict(torch.load(args.model))
    except:
        try:
            model.load_state_dict(torch.load(args.model)['model_state_dict'])
        except:
            later = True
    fixed_later = False
    try:
        fixed_model.load_state_dict(torch.load(args.fixed_model))
    except:
        try:
            fixed_model.load_state_dict(torch.load(args.fixed_model)['model_state_dict'])
        except:
            fixed_later = True

    model.cuda()
    model.train()
    fixed_model.cuda()
    if not args.fo_train:
        fixed_model.eval()
    return model,fixed_model,later,fixed_later

def set_dataset(args,root):
    if args.dataset == 'kitti':
        root = os.path.join(root,'KITTI')
        task_dist = datasets.KITTI({'crop_size': args.image_size, 
                                    'min_scale': 0, 
                                    'max_scale': 0, 
                                    'do_flip': False},root=root)
        eval_dist = datasets.KITTI(split='training',root=root)
    elif args.dataset == 'chairs':
        task_dist = datasets.FlyingChairs({'crop_size': [368,496], 
                                    'min_scale': 0, 
                                    'max_scale': 0, 
                                    'do_flip': False})
    elif args.dataset == 'sintel':
        root = os.path.join(root,'Sintel')
        task_dist = datasets.MpiSintel({'crop_size': args.image_size, 
                                    'min_scale': 0, 
                                    'max_scale': 0, 
                                    'do_flip': False},dstype=args.dstype,root=root)
        eval_dist = datasets.MpiSintel(dstype=args.dstype,root=root)
    return task_dist,eval_dist

def wrap_maml(args,models,laters):
    model,fixed_model = models
    later,fixed_later = laters
    if args.meta_algo == 'maml':
        #(1,3,288,496)
        model = l2l.algorithms.MAML(model,lr=args.inner_lr,allow_nograd=True,first_order=args.first_order) 
        fixed_model = l2l.algorithms.MAML(fixed_model,lr=args.inner_lr,allow_nograd=True,first_order=args.first_order) 

    elif args.meta_algo == 'metasgd':
        model = l2l.algorithms.MetaSGD(model,lr=args.inner_lr,first_order=args.first_order) 
        fixed_model = l2l.algorithms.MetaSGD(fixed_model,lr=args.inner_lr,first_order=args.first_order) 

    else:
        raise NotImplementedError

    if later:
        try:
            model.load_state_dict(torch.load(args.model)['model_state_dict'])
        except:
            model.module.load_state_dict(torch.load(args.model))
        model.cuda()
        model.train()
    if fixed_later:
        fixed_model.load_state_dict(torch.load(args.fixed_model)['model_state_dict'])
    fixed_model.cuda()

    if not args.fo_train:
        fixed_model.eval()

    return model,fixed_model 

def set_opt_loss(args,model,fixed_model):
    if args.optimizer == 'adam':
        opt = optim.Adam(model.parameters(),lr=args.lr) 
    elif args.optimizer == 'sgd':
        opt = optim.SGD(model.parameters(),lr=args.lr)
    else:
        raise ValueError("wrong optimizer")

    epoch=1
    if 'gma-' not in args.model:
        try:
            chkp = torch.load(args.model)['opt_state_dict']
            epoch = torch.load(args.model)['epoch']
            opt.load_state_dict(chkp)
        except:
            print("epoch and lr not loaded")
            pass

    if args.loss_fn == 'photometric':
        loss_fun = PhotoSmoothLoss(ptmodel=fixed_model,args=args) 
    elif args.loss_fn == 'arloss':
        loss_fun = unFlowLoss(args,fixed_model)
    else:
        loss_fun = None
    
    return opt, loss_fun, epoch

def main(args):
    root = './datasets'
    model,fixed_model,later,fixed_later = set_model(args)
    task_dist,eval_dist = set_dataset(args,root)
    model,fixed_model = wrap_maml(args,[model,fixed_model],[later,fixed_later])
    opt,loss_fun,epoch = set_opt_loss(args,model,fixed_model)
  
    print(args.dataset)
    print(args.validation)
    assert args.dataset == args.validation
    adaptation_indices, evaluation_indices = datasets.cross_samples(args.dataset,args.crossval,task_dist,args)

    if args.eval_idx:
        evaluation_indices = np.array(args.eval_idx)
    
    num_splits = len(adaptation_indices)//args.tasks_per_step

    wandb.init(config=args,name=str(args.name),group='maml',dir='/tmp')
    wandb.watch(model)
    if args.name == 'debug':
        os.environ['WANDB_MODE'] = 'offline'


    np.random.shuffle(adaptation_indices)
    adapt_split = np.array_split(adaptation_indices,num_splits)
    for i in range(epoch,EPOCHS+epoch): 
        if i % 10 ==1 and 'debug' not in args.name:
            validate_maml(model,evaluation_indices,eval_dist,i,args=args)             
        epe_steps=[]
        loss_steps=[]
        # images=[]
        # gts=[]
        step=0
        loss_terms = {}
        # uv_diff = []
        for piece in adapt_split:
            step_loss = 0.0
            for t in piece:
                data_blob = task_dist[t]
                # if args.dataset == 'kitti':
                    # image1, image2, flow, valid, gt_occ = [x[None].cuda() for x in data_blob]
                # else:
                image1, image2, flow, valid = [x[None].cuda() for x in data_blob[:-1]]
                learner = model.clone()
                for _ in range(args.inner_steps):
                    flow_pred = learner(image1=image1, image2=image2,iters=12) 
                    # if inner==0:
                        # uv_tic = flow_pred
                    # elif inner == args.inner_steps-1:
                        # uv_toc = flow_pred
                    if args.loss_fn == 'photometric':
                        loss,metrics = loss_fun(image1, image2, flow_pred,flow,valid,args.gamma,learner) #image1 (4,3,384,512)
                    elif args.loss_fn == 'L1':
                        loss, metrics = sequence_loss(flow_pred, flow, valid, args.gamma)
                    elif args.loss_fn == 'arloss':
                        flow_back = learner(image2,image1)
                        fb_flow = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flow_pred, flow_back)]
                        image_pair = torch.cat([image1, image2], 1) #(B,6,H,W)
                        loss,metrics,rangemap2 = loss_fun(fb_flow,image_pair,flow,valid,learner)
                        if args.name == 'debug':
                            for k,v in metrics.items():
                                if k not in loss_terms.keys():
                                    loss_terms[k]=0
                                loss_terms[k] += v
                    learner.adapt(loss,first_order=args.first_order) 

              
                flow_pred = learner(image1,image2) 
                adapt_loss,adapt_metrics = sequence_loss(flow_pred, flow, valid, args.gamma)
                step_loss += adapt_loss  
                step += 1
                print(f'{i}th epoch : {step} out of {num_splits} batches adapted')
                epe_steps.append(adapt_metrics['epe'])
                loss_steps.append(adapt_loss.item())

                if i % 500 ==1:
                    train_img = Image.fromarray(flow_img(flow_pred[-1])) 
                    save_scene_path = os.path.join(args.output,'train images')
                    if not os.path.exists(save_scene_path):
                        os.mkdir(save_scene_path)
                    save_scene_path = os.path.join(save_scene_path,f"epoch{i}_id{t}.png")
                    train_img.save(save_scene_path)

            step_loss = step_loss / args.tasks_per_step
            opt.zero_grad()
            step_loss.backward() 
            opt.step() 

            if args.fo_correct:
                with torch.no_grad():
                    pt_model = copy.deepcopy(model)
                pt_model.eval()
                loss_fun.pt_model = pt_model
      
        if args.name == 'debug':
            for k,v in loss_terms.items():
                print(loss_terms[k]/len(list(adaptation_indices)))
        save_dict = {'epoch' :i, 'model_state_dict':model.state_dict(),'opt_state_dict':opt.state_dict()}
        if i %50 ==0:
            torch.save(save_dict,os.path.join(args.output,f"step{i}.pth"))
    validate_maml(model,evaluation_indices,eval_dist,EPOCHS+1,args=args)

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
    parser.add_argument('--model_name')
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
    parser.add_argument('--num_steps', type=int, default=3000) #120000
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
    parser.add_argument('--loss_fn', default='arloss', help = 'photometric, L1, or arloss') 
    parser.add_argument('--one_image',type=int,default=-1,help='not for maml.py. keep it -1')
    parser.add_argument('--optimizer', type=str,default='adam', help="name your experiment")
    parser.add_argument('--occ_aware',default=False,help="if True, occlusion aware loss (continuous)")
    parser.add_argument('--eval_idx', type=int,default=[],nargs='+', help="it is okay to keep this empty")
    parser.add_argument('--fo_correct',default=False,help='set fo with each theta prime')
    parser.add_argument('--fo_train',default=False,help='set fo to be trainable')
    parser.add_argument('--convgru',default='',help="if count, aux losses for all gru iterations. else '' ")
    parser.add_argument('--inner_lr',default=0.000005,type=float) 
    parser.add_argument('--inner_steps',type=int,default=3)
    parser.add_argument('--first_order',default=True,type=str2bool)
    parser.add_argument('--no_orphan',default=False,type=str2bool)
    parser.add_argument('--force_size',default=False,help='avoid automatic image_size for datasets',type=str2bool)
    parser.add_argument('--image_size', type=int, nargs='+', default=[288, 960], help="sintel 368 768, kitti 288 960") 
    parser.add_argument('--fifo',default=True,help='|fi-fo| term',type=str2bool)
    parser.add_argument('--meta_algo',default='maml',help='maml, metasgd')
    parser.add_argument('--lr', type=float, default=0.000005)
    parser.add_argument('--fixed_model',default='',help='if not used, ''')

    parser.add_argument('--name', default='sintel_ml', help="name your experiment") #sintel 1/scene(10)
    parser.add_argument('--crossval',type=int,help="the index of pre-defined train/test split. you can try your own split", default=1)
    parser.add_argument('--model', help="restore checkpoint", default = './checkpoints/gma-things.pth')
    parser.add_argument('--dataset', help="dataset for meta-train", default="sintel")
    parser.add_argument('--output', type=str, default='./results/', help='output directory to save checkpoints and plots')
    parser.add_argument('--validation', type=str,default='sintel')
    parser.add_argument('--sm_scenes',type=str,nargs='+',default=['alley_1','ambush_5','bamboo_2','cave_4','mountain_1','sleeping_2','temple_2','market_2','shaman_2','bandage_1'], help="train test split by scenes")
    parser.add_argument('--num_per_scene',type=int,default=1, help="when train test split by scenes, how many samples per scene")
    parser.add_argument('--tasks_per_step',type=int,default=5)


    args = parser.parse_args()

    if args.no_orphan:
        assert args.meta_algo == 'metasgd'

    if args.fixed_model =='':
        args.fixed_model = args.model 

    remove_dir(args.output)
    args.output = os.path.join(args.output, str(args.name))
    if args.dataset == 'kitti':
        if not args.force_size:
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

    if 'gma-' not in args.model:
        try:
            args.lr = torch.load(args.model)['opt_state_dict']['param_groups'][0]['lr']
        except:
            print("lr not loaded")
            pass

    if args.name == 'debug':
        os.environ['WANDB_MODE'] = 'offline'
    main(args) 
