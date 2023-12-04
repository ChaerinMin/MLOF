import argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
# from torch_scatter import scatter_softmax, scatter_add
import os
import shutil

def remove_dir(log_dir):
    summary_list = os.listdir(log_dir)
    for i in summary_list:
        if 'debug' in i:
            shutil.rmtree(os.path.join(log_dir,i))
            print("{} has been deleted from {}".format(i,log_dir))
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel': 
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False,args=None): 
  
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    if args.first_order or args==None:
        img= F.grid_sample(img,grid,align_corners=True)
    else:
        img = grid_sample(img, grid, align_corners=True) #

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)

def coords_grid_y_first(batch, ht, wd):
    """Place y grid before x grid"""
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords, dim=0).int()
    return coords[None].expand(batch, -1, -1, -1)

def grid_sample(image, optical,align_corners=True):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val



def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow4(flow, mode='bilinear'):
    new_size = (4 * flow.shape[2], 4 * flow.shape[3])
    return 4 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow2(flow, mode='bilinear'):
    new_size = (2 * flow.shape[2], 2 * flow.shape[3])
    return 2 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def downflow8(flow, mode='bilinear'):
    new_size = (flow.shape[2] // 8, flow.shape[3] // 8)
    return F.interpolate(flow, size=new_size, mode=mode, align_corners=True) / 8


def downflow4(flow, mode='bilinear'):
    new_size = (flow.shape[2] // 4, flow.shape[3] // 4)
    return F.interpolate(flow, size=new_size, mode=mode, align_corners=True) / 4


def compute_interpolation_weights(yx_warped):
    # yx_warped: [N, 2]
    y_warped = yx_warped[:, 0]
    x_warped = yx_warped[:, 1]

    # elementwise operations below
    y_f = torch.floor(y_warped)
    y_c = y_f + 1
    x_f = torch.floor(x_warped)
    x_c = x_f + 1

    w0 = (y_c - y_warped) * (x_c - x_warped)
    w1 = (y_warped - y_f) * (x_c - x_warped)
    w2 = (y_c - y_warped) * (x_warped - x_f)
    w3 = (y_warped - y_f) * (x_warped - x_f)

    weights = [w0, w1, w2, w3]
    indices = [torch.stack([y_f, x_f], dim=1), torch.stack([y_c, x_f], dim=1),
               torch.stack([y_f, x_c], dim=1), torch.stack([y_c, x_c], dim=1)]
    weights = torch.cat(weights, dim=1)
    indices = torch.cat(indices, dim=2)
    # indices = torch.cat(indices, dim=0)  # [4*N, 2]

    return weights, indices

# weights, indices = compute_interpolation_weights(xy_warped, b, h_i, w_i)


def compute_inverse_interpolation_img(weights, indices, img, b, h_i, w_i):
    """
    weights: [b, h*w]
    indices: [b, h*w]
    img: [b, h*w, a, b, c, ...]
    """
    w0, w1, w2, w3 = weights
    ff_idx, cf_idx, fc_idx, cc_idx = indices

    k = len(img.size()) - len(w0.size())
    img_0 = w0[(...,) + (None,) * k] * img
    img_1 = w1[(...,) + (None,) * k] * img
    img_2 = w2[(...,) + (None,) * k] * img
    img_3 = w3[(...,) + (None,) * k] * img

    img_out = torch.zeros(b, h_i * w_i, *img.shape[2:]).type_as(img)

    ff_idx = torch.clamp(ff_idx, min=0, max=h_i * w_i - 1)
    cf_idx = torch.clamp(cf_idx, min=0, max=h_i * w_i - 1)
    fc_idx = torch.clamp(fc_idx, min=0, max=h_i * w_i - 1)
    cc_idx = torch.clamp(cc_idx, min=0, max=h_i * w_i - 1)

    img_out.scatter_add_(1, ff_idx[(...,) + (None,) * k].expand_as(img_0), img_0)
    img_out.scatter_add_(1, cf_idx[(...,) + (None,) * k].expand_as(img_1), img_1)
    img_out.scatter_add_(1, fc_idx[(...,) + (None,) * k].expand_as(img_2), img_2)
    img_out.scatter_add_(1, cc_idx[(...,) + (None,) * k].expand_as(img_3), img_3)

    return img_out  # [b, h_i*w_i, ...]
