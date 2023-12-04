# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image

# from core.utils.logger import flow_img

def plot_error(pred_flow,gt_flow):
    errors = np.linalg.norm(gt_flow - pred_flow, axis=-1) 
    errors[np.isinf(errors)] = 0
    errors[np.isnan(errors)] = 0
    errors_clip = np.clip(errors,0,7)
    errors = (errors * 255. / errors.max()).astype(np.uint8) 
    errors_clip = (errors_clip * 255. / errors_clip.max()).astype(np.uint8)
    errors = np.tile(errors[..., np.newaxis], [1, 1, 3]) 
    # errors[event_count_image == 0] = 0
    errors_clip = np.tile(errors_clip[..., np.newaxis], [1, 1, 3])
    return errors,errors_clip

def plot_error_diff(pred_occ,pred_normal, gt):
    errors_occ = np.linalg.norm(pred_occ - gt, axis=0) # 
    errors_normal = np.linalg.norm(pred_normal - gt, axis=0)  
    errors_occ[np.isinf(errors_occ)] = 0
    errors_occ[np.isnan(errors_occ)] = 0
    errors_normal[np.isinf(errors_normal)] = 0
    errors_normal[np.isnan(errors_normal)] = 0

    diff_pos = errors_occ-errors_normal
    diff_neg = -diff_pos
    diff_pos = np.clip(diff_pos,0,None)[...,np.newaxis]
    diff_neg = np.clip(diff_neg,0,None)[...,np.newaxis]
    diff = np.concatenate([diff_pos,diff_neg,diff_pos],axis=2)

    diff_clip = np.clip(diff,0,20)
    diff = (diff*255./diff.max()).astype(np.uint8)
    diff_clip = (diff_clip*255./20).astype(np.uint8) 
    
    return diff, diff_clip 


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    # if clip_flow is not None:
    #     rad = np.clip(rad, 0, clip_flow)
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]. numpy
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    if clip_flow is not None:
        rad = np.clip(rad, 0, clip_flow)
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def visu_jetcolor(range_map, gt_occ,clamp=[0.2,2]):

    range_map = torch.squeeze(range_map)
    gt_occ = torch.squeeze(gt_occ)
    occ_rm = torch.clone(range_map)
    noc_rm = torch.clone(range_map)

    
    occ_rm = np.clip(occ_rm.detach().cpu().numpy(),0,clamp[0])
    noc_rm = np.clip(noc_rm.detach().cpu().numpy(),0,clamp[1])

    occ_rm_colored = make_jetcolor(occ_rm)
    noc_rm_colored = make_jetcolor(noc_rm)
    occ_rm_colored = (occ_rm_colored*255./clamp[0]).astype(np.uint8)
    noc_rm_colored = (noc_rm_colored*255./clamp[1]).astype(np.uint8)
    occ_rm_colored[~gt_occ.bool().detach().cpu().numpy()] = 0.0 
    noc_rm_colored[gt_occ.bool().detach().cpu().numpy()] = 0.0  
    occ_rm_colored = Image.fromarray(occ_rm_colored)
    noc_rm_colored = Image.fromarray(noc_rm_colored)

    metrics = {
        'rangemap_occ' : occ_rm_colored,
        'rangemap_noc' : noc_rm_colored
    }
    return metrics

def make_jetcolor(img):
  
    cm = plt.get_cmap('jet')
    colored = cm(img) #input(h,w) output(h,w,4)
    colored = colored[...,:3]
    return colored


def determine_clip(flow,is_GT=True):
    flow = flow.permute([1,2,0]).detach().cpu().numpy()
    u = flow[:,:,0]
    v = flow[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    if is_GT:
        max_mag = np.max(rad)
        return max_mag
    else:
        mean = np.mean(rad)
        std = np.std(rad)
        return mean+2*std
