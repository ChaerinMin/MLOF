import sys

from core.utils.logger import error_map, flow_img, write_diff
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import imageio

from core.network import RAFTGMA

import core.datasets as datasets
from core.utils import flow_viz
from core.utils import frame_utils

from core.utils.utils import InputPadder, forward_interpolate
import wandb



@torch.no_grad()
def create_sintel_submission(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=32, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_sintel_submission_vis(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=32, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            # Visualizations
            flow_img = flow_viz.flow_to_image(flow)
            image = Image.fromarray(flow_img)
            if not os.path.exists(f'vis_test/RAFT/{dstype}/'):
                os.makedirs(f'vis_test/RAFT/{dstype}/flow')

            if not os.path.exists(f'vis_test/ours/{dstype}/'):
                os.makedirs(f'vis_test/ours/{dstype}/flow')

            if not os.path.exists(f'vis_test/gt/{dstype}/'):
                os.makedirs(f'vis_test/gt/{dstype}/image')

            # image.save(f'vis_test/ours/{dstype}/flow/{test_id}.png')
            image.save(f'vis_test/RAFT/{dstype}/flow/{test_id}.png')
            imageio.imwrite(f'vis_test/gt/{dstype}/image/{test_id}.png', image1[0].cpu().permute(1, 2, 0).numpy())
            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def create_kitti_submission_vis(model, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        # Visualizations
        flow_img = flow_viz.flow_to_image(flow)
        image = Image.fromarray(flow_img)
        if not os.path.exists(f'vis_kitti'):
            os.makedirs(f'vis_kitti/flow')
            os.makedirs(f'vis_kitti/image')

        image.save(f'vis_kitti/flow/{test_id}.png')
        imageio.imwrite(f'vis_kitti/image/{test_id}_0.png', image1[0].cpu().permute(1, 2, 0).numpy())
        imageio.imwrite(f'vis_kitti/image/{test_id}_1.png', image2[0].cpu().permute(1, 2, 0).numpy())


@torch.no_grad()
def validate_chairs(model, iters=6):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs_epe': epe}


@torch.no_grad()
def validate_things(model, iters=6):
    """ Perform evaluation on the FlyingThings (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        epe_list = []
        val_dataset = datasets.FlyingThings3D(dstype=dstype, split='validation')
        print(f'Dataset length {len(val_dataset)}')
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel(model, iters=6):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results

@torch.no_grad()
def validate_single_sintel(model, count=-1,iters=6,dstype='clean',args=None,wandb_save=True,root=None):
    model.eval()
    results = {}
    val_dataset = datasets.SintelSelect([args.one_image],split='training', dstype=dstype,root=root)
    epe_list = []

    image1, image2, flow_gt, valid_gt = val_dataset[0][:-1]
    image1 = image1[None].cuda()
    image2 = image2[None].cuda()

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
    flow = padder.unpad(flow_pr[0]).cpu()

    epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()

    epe = epe.view(-1)
    val = valid_gt.view(-1) >=0.5
    epe_list.append(epe[val].mean().item())
    epe_list = np.array(epe_list)

    # epe_list.append(epe.view(-1).numpy())
    # epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_list)

    if count >=0:
        iters_image = Image.fromarray(flow_img(flow,clip_flow=args.clip_flow))
        gt_image = Image.fromarray(flow_img(flow_gt,clip_flow=args.clip_flow))
        try:
            iters_image.save(os.path.join(args.save_pth,f"{count+1}.png"))
        except:
            iters_image.save(os.path.join(args.output,f"{count+1}.png"))
            gt_image.save(os.path.join(args.output,"gt.png"))

    if wandb_save:
        wandb.log({"val_epe":epe
                    # "val_flow":images,
                    # "val_gt":gts,
                    # "val_rgb":rgb
        })
    return {'kitti_epe':epe}

@torch.no_grad()
def validate_sintel_occ(model, iters=6):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['albedo', 'clean', 'final']:
    # for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
        epe_list = []
        epe_occ_list = []
        epe_noc_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _, occ, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            epe_noc_list.append(epe[~occ].numpy())
            epe_occ_list.append(epe[occ].numpy())

        epe_all = np.concatenate(epe_list)

        epe_noc = np.concatenate(epe_noc_list)
        epe_occ = np.concatenate(epe_occ_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        epe_occ_mean = np.mean(epe_occ)
        epe_noc_mean = np.mean(epe_noc)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Occ epe: %f, Noc epe: %f" % (epe_occ_mean, epe_noc_mean))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def separate_inout_sintel_occ():
    """ Peform validation using the Sintel (train) split """
    dstype = 'clean'
    val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)


    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, occ, occ_path = val_dataset[val_id]
        _, h, w = image1.size()
        coords = torch.meshgrid(torch.arange(h), torch.arange(w))
        coords = torch.stack(coords[::-1], dim=0).float()

        coords_img_2 = coords + flow_gt
        out_of_frame = (coords_img_2[0] < 0) | (coords_img_2[0] > w) | (coords_img_2[1] < 0) | (coords_img_2[1] > h)
        occ_union = out_of_frame | occ
        in_frame = occ_union ^ out_of_frame




@torch.no_grad()
def validate_kitti(model, logger=None, iters=6,length=200,args=None):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    gt_mag = []
    images=[]
    gts=[]

    for val_id in range(length): 
        image1, image2, flow_gt, valid_gt, gt_occ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()


        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()
        mag_mean = torch.mean(mag) 
        gt_mag.append(mag_mean.item())
        logger.write_flow(flow,flow_gt,val_id)

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item()) 
        out_list.append(out[val].cpu().numpy())
        images.append(wandb.Image(flow_img(flow[-1])))
        gts.append(wandb.Image(flow_img(flow_gt)))

    gt_mag = np.array(gt_mag)
    gt_mag_avg = np.mean(gt_mag) 
    if args:
        with open(os.path.join(args.output,"gt_mag.txt"),'w') as f:
            f.write(f"gt average magnitude : {gt_mag_avg:.4f}\n\n")
            f.write(" id   mag\n")
            for i in range(length):
                f.write(f"{i:^3} {gt_mag[i]:.4f}\n")

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    if logger is not None:
        logger.val_image_epe = epe_list
        logger.val_images = images
        logger.val_gt_images = gts

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti_epe': epe, 'kitti_f1': f1}

@torch.no_grad()
def validate_selected_kitti(model, logger=None, iters=6,length=200,args=None):
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    gt_mag = []
    images=[]
    gts=[]
    rgb=[]

    for val_id in args.val_list: 
        image1, image2, flow_gt, valid_gt, gt_occ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()


        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()
        mag_mean = torch.mean(mag) 
        gt_mag.append(mag_mean.item())
        logger.write_flow(flow,flow_gt,val_id)

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item()) 
        out_list.append(out[val].cpu().numpy())
        images.append(wandb.Image(flow_img(flow[-1])))
        gts.append(wandb.Image(flow_img(flow_gt)))
        rgb.append(wandb.Image(image1))

    gt_mag = np.array(gt_mag)
    gt_mag_avg = np.mean(gt_mag) 
    if args:
        with open(os.path.join(args.output,"gt_mag.txt"),'w') as f:
            f.write(f"gt average magnitude : {gt_mag_avg:.4f}\n\n")
            f.write(" id   mag\n")
            for i in range(length):
                f.write(f"{i:^3} {gt_mag[i]:.4f}\n")

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    if logger is not None:
        logger.val_image_epe = epe_list
        logger.val_images = images
        logger.val_gt_images = gts

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti_epe': epe, 'kitti_f1': f1,"flow_pred":images,"gts":gts,"rgb":rgb}


@torch.no_grad()
def evaluate_single_kitti(model, logger=None, iters=6,length=1,args=None):
  
    val_dataset = datasets.SingleKITTI(args.one_image,split='training')
    image1, image2, flow_gt, valid_gt, gt_occ = val_dataset[0]
    image1 = image1[None].cuda()
    image2 = image2[None].cuda()
    padder = InputPadder(image1.shape, mode='kitti')
    image1, image2 = padder.pad(image1, image2)
    val = valid_gt >= 0.5
    val = np.tile(val[np.newaxis,np.newaxis,...], [1,2,1, 1]) 

    model.eval()
    try:
        model.load_state_dict(torch.load(args.compare_err[0]))
    except:
        model.load_state_dict(torch.load(args.compare_err[0])['model_state_dict'])
    _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
    flow_occ = padder.unpad(flow_pr[0]).cpu()

    try:
        model.load_state_dict(torch.load(args.compare_err[1]))
    except:
        model.load_state_dict(torch.load(args.compare_err[1])['model_state_dict'])
    _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
    flow_normal = padder.unpad(flow_pr[0]).cpu()
    
    flow_occ[val] =0 
    flow_normal[val] =0 
    flow_gt[val] =0 

    diff, diff_clip = flow_viz.plot_error_diff(flow_occ,flow_normal, flow_gt)
    write_diff(diff,diff_clip,args)

@torch.no_grad()
def validate_single_kitti(model,count=-1, logger=None, iters=6,length=1,args=None,id=None,wandb_save=True,root=None):

    images=[]
    gts=[]
    rgb=[]
    err=[]
    err_clip=[]
    model.eval()
    if id is not None:
        val_dataset = datasets.SingleKITTI(id,split='training',root=root)
    else:
        val_dataset = datasets.SingleKITTI(args.one_image,split='training',root=root)

    out_list, epe_list = [], []
    gt_mag = []
    image1, image2, flow_gt, valid_gt, gt_occ = val_dataset[0]
    image1 = image1[None].cuda()
    image2 = image2[None].cuda()

    padder = InputPadder(image1.shape, mode='kitti')
    image1, image2 = padder.pad(image1, image2)

    _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
    flow = padder.unpad(flow_pr[0]).cpu()

    if logger is not None:
        logger.write_flow(flow,flow_gt,args.one_image)

    epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
    mag = torch.sum(flow_gt**2, dim=0).sqrt() 
    mag_mean = torch.mean(mag) 
    gt_mag.append(mag_mean.item())

    epe = epe.view(-1)
    mag = mag.view(-1)
    val = valid_gt.view(-1) >= 0.5

    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
    epe_list.append(epe[val].mean().item())
    out_list.append(out[val].cpu().numpy())

    gt_mag = np.array(gt_mag)
    gt_mag_avg = np.mean(gt_mag) 


    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    images.append(wandb.Image(flow_img(flow)))
    gts.append(wandb.Image(flow_img(flow_gt)))
    rgb.append(wandb.Image(image1))
    er, er_clip = error_map(flow,flow_gt)
    err.append(wandb.Image(er))
    err_clip.append(wandb.Image(er_clip))

    if count %3==0:
        iters_image = Image.fromarray(flow_img(flow))
        iters_image.save(os.path.join(args.save_pth,f"img{args.one_image}_iters{count}.png"))
        if count>0:
            save_dict = {'count' :count, 'model_state_dict':model.state_dict()}
            torch.save(save_dict,os.path.join(args.save_pth,f"img{args.one_image}_iters{count}.pth"))

    if logger is not None:
        logger.val_image_epe = epe_list
        logger.val_images = images
        logger.val_gt_images = gts
        logger.val_rgb = rgb
        logger.val_err = err
        logger.val_err_clip = err_clip

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    # print("Validation KITTI: %f, %f" % (epe, f1))
    if wandb_save:
        wandb.log({"val_epe":epe
                    # "val_flow":images,
                    # "val_gt":gts,
                    # "val_rgb":rgb
        },count)
    
    return {'kitti_epe': epe, 'kitti_f1': f1,"flow_pred":images,"gts":gts,"rgb":rgb,'gt_mag':gt_mag_avg}

def uni_validate(args,model,count,iters,root):
    if args.validation == 'kitti':
        return validate_single_kitti(model,count=count,iters=iters,args=args,root=root)
    elif args.validation == 'sintel':
        return validate_single_sintel(model,count=count,iters=iters,dstype=args.dstype,args=args,wandb_save=True,root=root)
    else: 
        raise NotImplementedError

'''Uncomment below only for debuging purposes'''

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', help="restore checkpoint", default = 'checkpoints/gma-chairs.pth')
#     parser.add_argument('--dataset', help="dataset for evaluation",default = 'kitti') 
#     parser.add_argument('--iters', type=int, default=12)
#     parser.add_argument('--num_heads', default=1, type=int,
#                         help='number of heads in attention and aggregation')
#     parser.add_argument('--position_only', default=False, action='store_true',
#                         help='only use position-wise attention')
#     parser.add_argument('--position_and_content', default=False, action='store_true',
#                         help='use position and content-wise attention')
#     parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
#     parser.add_argument('--model_name')

#     # Ablations
#     parser.add_argument('--replace', default=False, action='store_true',
#                         help='Replace local motion feature with aggregated motion features')
#     parser.add_argument('--no_alpha', default=False, action='store_true',
#                         help='Remove learned alpha, set it to 1')
#     parser.add_argument('--no_residual', default=False, action='store_true',
#                         help='Remove residual connection. Do not add local features with the aggregated features.')

#     args = parser.parse_args()

#     if args.dataset == 'separate':
#         separate_inout_sintel_occ()
#         sys.exit()

#     model = torch.nn.DataParallel(RAFTGMA(args))
#     model.load_state_dict(torch.load(args.model))

#     model.cuda()
#     model.eval()



#     with torch.no_grad():
#         if args.dataset == 'chairs':
#             validate_chairs(model.module, iters=args.iters)

#         elif args.dataset == 'things':
#             validate_things(model.module, iters=args.iters)

#         elif args.dataset == 'sintel':
#             validate_sintel(model.module, iters=args.iters)

#         elif args.dataset == 'sintel_occ':
#             validate_sintel_occ(model.module, iters=args.iters)

#         elif args.dataset == 'kitti':
#             validate_kitti(model.module, iters=args.iters)
