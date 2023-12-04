import inspect
import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.nn as nn

from core.utils.utils import grid_sample
MAX_FLOW = 400
def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid

def flow_warp(x, flow12, pad='border', mode='bilinear',args=None): #ARflow
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(grid_sample).args:
        if args.first_order or args == None:
            im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
        else:
            im1_recons = grid_sample(x, v_grid)
    else:
        if args.first_order or args == None:
            im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
        else:
            im1_recons = grid_sample(x, v_grid)
        
    return im1_recons

def warp_images_with_flow(images, flow):
    """
    Generates a prediction of an image given the optical flow, as in Spatial Transformer Networks.
    """
    dim3 = 0
    if images.dim() == 3:
        dim3 = 1
        images = images.unsqueeze(0)
        flow = flow.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]
    flow_x,flow_y = flow[:,1,...],flow[:,0,...]
    coord_x, coord_y = torch.meshgrid(torch.arange(height), torch.arange(width))

    pos_x = coord_x.reshape(height,width).type(torch.float32).cuda() + flow_x
    pos_y = coord_y.reshape(height,width).type(torch.float32).cuda() + flow_y
    pos_x = (pos_x-(height-1)/2)/((height-1)/2)
    pos_y = (pos_y-(width-1)/2)/((width-1)/2)

    pos = torch.stack((pos_y,pos_x),3).type(torch.float32) #(1,384,512,2)
    result = torch.nn.functional.grid_sample(images, pos, mode='bilinear', padding_mode='zeros',align_corners=True) #(1,3,384,512)
    if dim3 == 1:
        result = result.squeeze()
        
    return result

def get_grid(batch_size, H, W, start):
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    ones = torch.ones_like(xx)
    grid = torch.cat((xx, yy, ones), 1).float()
    if torch.cuda.is_available():
        grid = grid.cuda()
    # print("grid",grid.shape)
    # print("start", start)
    grid[:, :2, :, :] = grid[:, :2, :, :] + start 

    return grid  

def transformer(I, vgrid, train=True):
    # I: Img, shape: batch_size, 1, full_h, full_w
    # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
    # outsize: (patch_h, patch_w)

    def _repeat(x, n_repeats):

        rep = torch.ones([n_repeats, ]).unsqueeze(0)
        rep = rep.int()
        x = x.int()

        x = torch.matmul(x.reshape([-1, 1]), rep)
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size, scale_h):
        # x: x_grid_flat
        # y: y_grid_flat
        # out_size: same as im.size
        # scale_h: True if normalized
        # constants
        num_batch, num_channels, height, width = im.size()

        out_height, out_width = out_size[0], out_size[1]
        # zero = torch.zeros_like([],dtype='int32')
        zero = 0
        max_y = height - 1
        max_x = width - 1
        if scale_h:
      
            x = (x + 1.0) * (height) / 2.0
            y = (y + 1.0) * (width) / 2.0

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        dim1 = torch.from_numpy(np.array(width * height))
        dim2 = torch.from_numpy(np.array(width))

        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width) 
       
        if torch.cuda.is_available():
            dim2 = dim2.cuda()
            dim1 = dim1.cuda()
            y0 = y0.cuda()
            y1 = y1.cuda()
            x0 = x0.cuda()
            x1 = x1.cuda()
            base = base.cuda()
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _transform(I, vgrid, scale_h):

        C_img = I.shape[1]
        B, C, H, W = vgrid.size()

        x_s_flat = vgrid[:, 0, ...].reshape([-1])
        y_s_flat = vgrid[:, 1, ...].reshape([-1])
        out_size = vgrid.shape[2:]
        input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size, scale_h)

        output = input_transformed.reshape([B, H, W, C_img])
        return output

    # scale_h = True
    output = _transform(I, vgrid, scale_h=False)
    if train:
        output = output.permute(0, 3, 1, 2)
    return output

def warp_im(I_nchw, flow_nchw):
    start = np.zeros((1, 2, 1, 1)) 
    start = torch.from_numpy(start).float().cuda()

    batch_size, _, img_h, img_w = I_nchw.size()
    _, _, patch_size_h, patch_size_w = flow_nchw.size()
    patch_indices = get_grid(batch_size, patch_size_h, patch_size_w, start) 
    vgrid = patch_indices[:, :2, ...]
    # grid_warp = vgrid - flow_nchw
    grid_warp = vgrid + flow_nchw
    pred_I2 = transformer(I_nchw, grid_warp) 
    return pred_I2


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    """
    Robust Charbonnier loss, as defined in equation (4) of the paper.
    """
    loss = torch.mean(torch.pow((delta ** 2 + epsilon ** 2), alpha))
    return loss


def compute_smoothness_loss(flow): 
    """ 
    Local smoothness loss, as defined in equation (5) of the paper.
    The neighborhood here is defined as the 8-connected region around each pixel.
    """
    #배치는 한 번에
    flow_ucrop = flow[..., 1:]
    flow_dcrop = flow[..., :-1]
    flow_lcrop = flow[..., 1:, :]
    flow_rcrop = flow[..., :-1, :]

    flow_ulcrop = flow[..., 1:, 1:]
    flow_drcrop = flow[..., :-1, :-1]
    flow_dlcrop = flow[..., :-1, 1:]
    flow_urcrop = flow[..., 1:, :-1]
    
    smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) +\
                      charbonnier_loss(flow_ucrop - flow_dcrop) +\
                      charbonnier_loss(flow_ulcrop - flow_drcrop) +\
                      charbonnier_loss(flow_dlcrop - flow_urcrop)
    smoothness_loss /= 4.
    
    return smoothness_loss


def compute_photometric_loss(prev_images, next_images, flow,valid,args=None): 
    batch_photometric_loss = 0.
    batch_size = prev_images.shape[0]
    for image_num in range(batch_size): 
        flow_pred = flow[image_num]
        height = flow_pred.shape[1]
        width = flow_pred.shape[2]
        
        prev_images_resize = F.to_tensor(F.resize(F.to_pil_image(prev_images[image_num].cpu()),
                                                [height, width])).cuda()   #(3,384,512)
        next_images_resize = F.to_tensor(F.resize(F.to_pil_image(next_images[image_num].cpu()), 
                                                [height, width])).cuda()
        next_images_warped = flow_warp(torch.unsqueeze(next_images_resize,dim=0),torch.unsqueeze(flow_pred,0),args=args)
        distance = next_images_warped - prev_images_resize
        distance = valid[image_num]*distance
        photometric_loss = charbonnier_loss(distance) 
        batch_photometric_loss += photometric_loss

    batch_photometric_loss /= batch_size
    return batch_photometric_loss

class PhotoSmoothLoss(torch.nn.Module):
    def __init__(self,ptmodel, args, smoothness_weight=0.5):
        super(PhotoSmoothLoss, self).__init__()
     
        self.pt_model = ptmodel
        self._smoothness_weight = smoothness_weight
        self._wkeeper = args.wkeeper
        self.args=args
        if not args.loss_iters:
            self.start_iter = args.iters - 1
        else:
            self.start_iter=0

    def forward(self,prev_images, next_images, flow,flow_gt,valid,gamma,model):  
        """
        Multi-scale photometric loss, as defined in equation (3) of the paper.
        """

        weight_reg = 0.0 
        
        for pt, ft in zip(self.pt_model.named_parameters(),model.named_parameters()):
            if pt[0] != ft[0]:
                raise ValueError("parameter name not matched")
            # weight_len += pt[1].numel()
            tmp = torch.sum(torch.pow(pt[1]-ft[1],2))
            weight_reg = weight_reg + tmp

        n_predictions = len(flow)  
        flow_loss = 0.0
        total_photo_loss = 0.0
        total_smooth_loss = 0.0
        valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < MAX_FLOW) #(4,368,496) True/False values
        for i in range(self.start_iter,n_predictions):
            i_weight = gamma**(n_predictions - i - 1) 
            photometric_loss = compute_photometric_loss(prev_images, next_images, flow[i],valid,self.args)
            smoothness_loss = compute_smoothness_loss(flow[i])
            photo_loss = i_weight*photometric_loss
            smooth_loss = i_weight*self._smoothness_weight*smoothness_loss
            flow_loss += photo_loss + smooth_loss
            total_photo_loss += photo_loss
            total_smooth_loss += smooth_loss
        total_loss = flow_loss + self._wkeeper*weight_reg 

        epe = torch.sum((flow[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]

        metrics = {
       
            'epe': epe.mean().item(),
            # '1px': (epe < 1).float().mean().item(),
            # '3px': (epe < 3).float().mean().item(),
            # '5px': (epe < 5).float().mean().item(),
        }

        return total_loss, metrics


