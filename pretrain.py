from __future__ import print_function, division
import sys

import wandb

from core.utils.logger import Logger, flow_img
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

from core.network import RAFTGMA

from core.utils import flow_viz
import core.datasets as datasets
import evaluate

from torch.cuda.amp import GradScaler

# exclude extremly large displacements
MAX_FLOW = 400


def convert_flow_to_image(image1, flow):
    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)
    flow_image = cv2.resize(flow_image, (image1.shape[3], image1.shape[2]))
    return flow_image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_loss(flow_preds, flow_gt, valid, gamma): 
    
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < MAX_FLOW) #(4,368,496) True/False values

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1) 
        i_loss = (flow_preds[i] - flow_gt).abs() 
        flow_loss += i_weight * (valid[:, None] * i_loss).mean() 

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt() 
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model,schedule = 'cycle'):
    """ Create the optimizer and learning rate scheduler """
    if args.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.wdecay)
    else:
        raise ValueError("wrong optimizer.")

    if schedule == 'cycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    elif schedule == None:
        scheduler = None
    else:
        raise ValueError

    return optimizer, scheduler


def main(args):

    model = nn.DataParallel(RAFTGMA(args), device_ids=args.gpus)

    print(f"Parameter Count: {count_parameters(model)}")

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    # if args.stage != 'chairs':
    #     model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args)

    while logger.total_steps <= args.num_steps:
        train(model, train_loader, optimizer, scheduler, logger, scaler, args)
        if logger.total_steps >= args.num_steps:
            # plot_train(logger, args)
            # plot_val(logger, args)
            break

    PATH = args.output+f'/{args.name}.pth'
    torch.save(model.state_dict(), PATH)
    return PATH


def train(model, train_loader, optimizer, scheduler, logger, scaler, args):
    rgb=[]
    images=[]
    gts=[]
    for i_batch, data_blob in enumerate(train_loader):
        tic = time.time()
        image1, image2, flow, valid, gt_occ = [x.cuda() for x in data_blob]

        optimizer.zero_grad()

        flow_pred = model(image1, image2)

        loss, metrics = sequence_loss(flow_pred, flow, valid, args.gamma)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        toc = time.time()
        print(f"{i_batch}/{len(train_loader)} trained")

        metrics['time'] = toc - tic
        if logger.total_steps % logger.args.print_freq ==logger.args.print_freq-1 or logger.total_steps == 0: 
            rgb.append(wandb.Image(image1[0]))
            images.append(wandb.Image(flow_img(flow_pred[-1])))
            gts.append(wandb.Image(flow_img(flow)))
        logger.push(metrics,images,gts,rgb)

        # Validate
        if logger.total_steps % args.val_freq == args.val_freq - 1:
            validate(model, args, logger)
            # plot_train(logger, args)
            # plot_val(logger, args)
            PATH = args.output + f'/{logger.total_steps+1}_{args.name}.pth'
            torch.save(model.state_dict(), PATH)
            print("validation done")

        if logger.total_steps >= args.num_steps:
            break


def validate(model, args, logger):
    model.eval()
    results = {}

    # Evaluate results
    for val_dataset in args.validation:
        if val_dataset == 'chairs':
            results.update(evaluate.validate_chairs(model.module, args.iters)) 
        elif val_dataset == 'sintel':
            results.update(evaluate.validate_sintel(model.module, args.iters))
        elif val_dataset == 'kitti':
            # if args.val_list:
            #     results.update(evaluate.validate_selected_kitti(model.module,args.iters))
            # else:
            results.update(evaluate.validate_kitti(model.module, args.iters))

    # Record results in logger
    for key in results.keys():
        if key not in logger.val_results_dict.keys():
            logger.val_results_dict[key] = []
        logger.val_results_dict[key].append(results[key])

    logger.val_steps_list.append(logger.total_steps)
    model.train()


def plot_val(logger, args):
    for key in logger.val_results_dict.keys():
        # plot validation curve
        plt.figure()
        plt.plot(logger.val_steps_list, logger.val_results_dict[key])
        plt.xlabel('x_steps')
        plt.ylabel(key)
        plt.title(f'Results for {key} for the validation set')
        plt.savefig(args.output+f"/{key}.png", bbox_inches='tight')
        plt.close()


def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_epe_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output+"/train_epe.png", bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='gma-chairs', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training",default='chairs')
    parser.add_argument('--validation', type=str, nargs='+',default='chairs')
    parser.add_argument('--restore_ckpt', help="restore checkpoint",default=None) 
    parser.add_argument('--output', type=str, default='./results/pretrained/', help='output directory to save checkpoints and plots') 

    parser.add_argument('--lr', type=float, default=0.000125)
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, nargs='+', default=[368, 496], help='chairs 368 496, things 400 720, sintel 368 768, kitti 288 960')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])

    parser.add_argument('--wdecay', type=float, default=0.0001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=12, help='GMA(RAFT) GRU iterations')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='printing frequency')

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--model_name', default='', help='not for train.py keep it empty')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--id_list', type=int, nargs='+', default=[], help="theta^{ft}_n")
    parser.add_argument('--val_list', type=int, nargs='+', default=[], help="when trainig theta^{ft}_n, use this to validate")
    parser.add_argument('--one_image',type=int,default=-1,help='not for train.py. keep it -1')    

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    main(args)
