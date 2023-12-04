import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.advanced_loss.warp_utils import get_corresponding_map

from core.losses import mesh_grid


# Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = image[:, 0, :, :] * 0.2989 + \
                    image[:, 1, :, :] * 0.5870 + \
                    image[:, 2, :, :] * 0.1140
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask


def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, 1)(x) 
    mu_y = nn.AvgPool2d(patch_size, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy

def positive_exp(data):
    result = -torch.exp(-data)+1
    return result

def smooth_grad_1st(flo, image, alpha):
    img_dx, img_dy = gradient(image) 
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2

    return loss_x.mean() / 2. + loss_y.mean() / 2.


def smooth_grad_2nd(flo, image, alpha):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def smooth_grad_1st_occ_awr(flo,image,range_map,alpha):
    raise NotImplementedError

def smooth_grad_2nd_occ_awr(flo,image,range_map,alpha):
    img_dx, img_dy = gradient(image) 
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
    weights_occ = torch.exp(-range_map*100) 

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * weights_occ[:,:,:,2:]*dx2.abs() 
    loss_y = weights_y[:, :, 1:, :] * weights_occ[:,:,2:,:]*dy2.abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.


def occ_accuracy_hard(est_occ,gt_occ):

    est_occ = 1-est_occ 
    diff = gt_occ == est_occ 
    accuracy = torch.sum(diff) / torch.numel(gt_occ)
    return accuracy 

def occ_accuracy_soft(range_map,gt_occ):
  
    gt_noc = 1-gt_occ
    gt_occ, gt_noc = gt_occ.type(torch.bool), gt_noc.type(torch.bool)

    occ_score = range_map[0].view(-1)[gt_occ.view(-1)] 
    if occ_score.nelement() == 0:
        occ_score = np.NaN
    else:
        occ_score = torch.mean(occ_score).item() #the less the better (non-negative)

    noc_score = range_map[0].view(-1)[gt_noc.view(-1)] 
    if noc_score.nelement() == 0:
        noc_score = np.NaN
    else:
        noc_score = torch.mean(noc_score).item() #the greater the better (non-negative)

    rangemap_metrics = {
        'occ_score' : occ_score,
        'noc_score' : noc_score
    }
    return rangemap_metrics