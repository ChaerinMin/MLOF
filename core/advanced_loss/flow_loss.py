import torch
import torch.nn as nn
import torch.nn.functional as F
from core.advanced_loss.loss_blocks import SSIM, occ_accuracy_hard, occ_accuracy_soft, positive_exp, smooth_grad_1st, smooth_grad_1st_occ_awr, smooth_grad_2nd, TernaryLoss, smooth_grad_2nd_occ_awr
from core.advanced_loss.warp_utils import flow_warp
from core.advanced_loss.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward
MAX_FLOW = 400

class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg, ptmodel):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg
        self.pt_model = ptmodel
        self.meta_algo = cfg.meta_algo 
        self.fifo = cfg.fifo 


    def loss_photo_regular(self, im1_scaled, im1_recons, occu_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1,
                                            im1_scaled * occu_mask1)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                      im1_scaled * occu_mask1)]

        return sum([l.mean() for l in loss]) / occu_mask1.mean()
    
    def loss_photo_occ_awr(self,im1_scaled,im1_recons,range_map,alpha):#
        weights = positive_exp(range_map * 0.01) 
      
        loss =[]
        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs()]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons,
                                            im1_scaled)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons,
                                                      im1_scaled)]
        loss = [weights*l for l in loss] 

        return sum([l.mean() for l in loss])#
       

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.alpha)]
        return sum([l.mean() for l in loss])

    def loss_smooth_occ_awr(self, flow, im1_scaled,range_map):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd_occ_awr
        else:
            func_smooth = smooth_grad_1st_occ_awr
        loss = []
        loss += [func_smooth(flow, im1_scaled, range_map,self.cfg.alpha)]
        return sum([l.mean() for l in loss])

    def forward(self, output, target,gt,valid,model,gt_occ=None):
        """
        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :param gt: (B,2,H,W)
        :return:
        """
        weight_reg = 0.0 
        # if self.meta_algo == 'maml':
        self.pt_model.eval()
        if self.fifo:
            for pt, ft in zip(self.pt_model.named_parameters(),model.named_parameters()): 
              
                if pt[0] != ft[0]:
                    raise ValueError("파라미터 이름 불일치")
                tmp = torch.sum(torch.pow(pt[1]-ft[1],2))
                weight_reg = weight_reg + tmp

        pyramid_flows = output
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]

        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []
        valid = (valid >= 0.5) & ((gt**2).sum(dim=1).sqrt() < MAX_FLOW) #(4,368,496) True/False values
        
        s = 1.
        if self.cfg.convgru == 'count':
            flag = 0
        elif self.cfg.convgru == '':
            flag = 11
        else:
            raise ValueError

        for i, flow in enumerate(pyramid_flows):
            if not self.cfg.convgru:
                if self.cfg.w_scales[i] == 0:
                    pyramid_warp_losses.append(0)
                    pyramid_smooth_losses.append(0)
                    continue

            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad,args=self.cfg)
            im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad,args=self.cfg)

            if i == flag: 
                if self.cfg.occ_from_back: #True
                    occu_mask1, range_map1 = get_occu_mask_backward(flow[:, 2:], th=0.2) 
                    occu_mask1 = 1-occu_mask1 #

                    occu_mask2, range_map2 = get_occu_mask_backward(flow[:, :2], th=0.2)

                    occu_mask2 = 1-occu_mask2
                else:
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:],args=self.cfg)
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2],args=self.cfg)
            else:
                occu_mask1 = F.interpolate(self.pyramid_occu_mask1[0],
                                           (h, w), mode='nearest')
                occu_mask2 = F.interpolate(self.pyramid_occu_mask2[0],
                                           (h, w), mode='nearest')

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            if self.cfg.occ_aware:
                loss_warp = self.loss_photo_occ_awr(im1_scaled,im1_recons,range_map1,self.cfg.alpha)
                loss_smooth = self.loss_smooth_occ_awr(flow[:, :2] / s, im1_scaled,range_map1)
            else:
                loss_warp = self.loss_photo_regular(im1_scaled, im1_recons, occu_mask1) 
                loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_scaled)
             

            if i == flag: 
                s = min(h, w)


            if self.cfg.with_bk: #True. 
                if self.cfg.occ_aware:
                    loss_warp += self.loss_photo_occ_awr(im2_scaled,im2_recons,range_map2,self.cfg.alpha)
                    loss_smooth += self.loss_smooth_occ_awr(flow[:, 2:] / s, im2_scaled,range_map2)
                else:
                    loss_warp += self.loss_photo_regular(im2_scaled, im2_recons,
                                                   occu_mask2)
                    loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_scaled)

                loss_warp /= 2.
                loss_smooth /= 2.

            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)

        if not self.cfg.convgru:
            pyramid_warp_losses = [l * w for l, w in
                                zip(pyramid_warp_losses, self.cfg.w_scales)]
            pyramid_smooth_losses = [l * w for l, w in
                                    zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]
            warp_loss = sum(pyramid_warp_losses)
            smooth_loss = self.cfg.w_smooth * sum(pyramid_smooth_losses) 
        elif self.cfg.convgru == 'count':                       
            n_predictions = len(pyramid_warp_losses) #12
            warp_loss = 0
            smooth_loss = 0
            for i in range(n_predictions):
                i_weight = self.cfg.gamma**(n_predictions-i-1) 
                warp_loss += i_weight*pyramid_warp_losses[i]
                smooth_loss += i_weight*pyramid_smooth_losses[i]

            smooth_loss = self.cfg.w_smooth*smooth_loss
        total_loss = 4*warp_loss + smooth_loss + self.cfg.wkeeper*weight_reg 

        epe = torch.sum((output[-1][:,:2] - gt)**2, dim=1).sqrt()  
        epe = epe.view(-1)[valid.view(-1)] 

        if isinstance(weight_reg,torch.Tensor):
            weight_reg = weight_reg.item()
            
        metrics = {
            'loss_total': total_loss.item(),
            'loss_photo' : warp_loss.item(),
            'loss_smooth' : smooth_loss.item(),
            'loss_params_change' : weight_reg,
            'epe': epe.mean().item()
            # '1px': (epe < 1).float().mean().item(),
            # '3px': (epe < 3).float().mean().item(),
            # '5px': (epe < 5).float().mean().item(),
        }

        if gt_occ is not None:
            occ_acc_hard = occ_accuracy_hard(occu_mask2,gt_occ) 
            range_map_metrics = occ_accuracy_soft(range_map2,gt_occ)
            metrics.update(range_map_metrics)
            metrics['occ_accuracy_hard'] = occ_acc_hard.item()

        # return total_loss, warp_loss, smooth_loss, pyramid_flows[0].abs().mean()
        return total_loss, metrics, range_map2

def occ_loss_binary(flow_pred,gt_occ):
    occu_mask, range_map = get_occu_mask_backward(flow_pred[:,2:], th=0.2)  
    loss = F.binary_cross_entropy(occu_mask,gt_occ)
    metrics = {
        'occ_learn' : loss.item()
    }
    return loss, metrics