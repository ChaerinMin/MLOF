import os
import numpy as np
import torch
from PIL import Image
from core.utils.logger import alto_visu

import evaluate
import wandb
from core.utils.logger import flow_img
from core.utils.utils import InputPadder

'''
validate.py does not have an entry. validate.py only works as a module.
'''

def identify_dataset(sample,args):
    assert sample.shape[-1]>3
    mode = ''
    if sample.shape[-1] == 496:
        mode = 'chairs'
    elif args.dataset == 'sintel':
        mode = 'sintel'
    elif sample.shape[-1] == 960:
        mode = 'kitti'
    elif args.crossval>3:
        if args.dataset == 'kitti':
            mode = 'kitti'
        elif args.dataset == 'sintel':
            mode = 'sintel'
    else:
        raise ValueError
    return mode



def validate_maml(model,evaluation_indices,eval_dist,i,args=None,optim=None):
    model.eval()
    val_epes = []
    scene_epes = {}
    if isinstance(eval_dist,torch.utils.data.Dataset):
        for s,t in enumerate(evaluation_indices):
            try:
                image1,image2,flow,valid,gt_occ = eval_dist[t] 
            except:
                image1,image2,flow,valid = eval_dist[t] 
            extra_info = eval_dist.extra_info[t]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()


            mode = identify_dataset(image1,args)
            padder = InputPadder(image1.shape, mode=mode) 
            image1, image2 = padder.pad(image1, image2)
            _, flow_pr = model(image1=image1, image2=image2, iters=12, test_mode=True)
            val_pred = padder.unpad(flow_pr[0]).cpu()
            
            epe = torch.sum((val_pred - flow)**2, dim=0).sqrt()
            epe = epe.view(-1)
            val = valid.view(-1) >= 0.5
            valid_epe = epe[val].mean().item()
            val_epes.append(valid_epe) 
            if args.dataset == 'sintel':
                scene_epes[extra_info[0]] = scene_epes.get(extra_info[0],[])+[valid_epe]
            if i % 500 ==1:
                if args.dataset == 'sintel':
                    if extra_info[1] %5 ==0:
                        scene_img = Image.fromarray(flow_img(val_pred))
                        save_scene_path = os.path.join(args.output,extra_info[0])
                        if not os.path.exists(save_scene_path):
                            os.mkdir(save_scene_path)
                        save_scene_path = os.path.join(save_scene_path,f"epoch{i}_id{t}.png")
                        scene_img.save(save_scene_path)
                elif args.dataset == 'kitti':
                    if t%3 ==0:
                        val_img = Image.fromarray(flow_img(val_pred))
                        save_val_path = os.path.join(args.output,"validation images")
                        if not os.path.exists(save_val_path):
                            os.mkdir(save_val_path)
                        save_val_path = os.path.join(save_val_path,f"epoch{i}_id{t}.png")
                        val_img.save(save_val_path)
                else:
                    raise NotImplementedError

            print(s,sep=" ")
    elif isinstance(eval_dist,torch.utils.data.DataLoader):
        for k,data_blob in enumerate(eval_dist):
            image1,image2,flow,valid = data_blob[:-1] 
            image1 = image1.cuda()
            image2 = image2.cuda()
            flow = torch.squeeze(flow)
            valid = torch.squeeze(valid)


            mode = identify_dataset(image1)
            padder = InputPadder(image1.shape, mode=mode) 
            image1, image2 = padder.pad(image1, image2)
            _, flow_pr = model(image1, image2, iters=12, test_mode=True)
            val_pred = padder.unpad(flow_pr[0]).cpu()
            epe = torch.sum((val_pred - flow)**2, dim=0).sqrt()
            epe = epe.view(-1)
            val = valid.view(-1) >= 0.5
            val_epes.append(epe[val].mean().item()) 
            print(f"validation : {k}/{len(eval_dist)}  EPE {val_epes[0]:.3f}")
    else:
        raise ValueError
    
    val_aee = np.mean(val_epes)
    # val_std = np.std(val_epes)
    if optim is None:
        wandb.log({'val_epe':val_aee,
                    # 'val_std':val_std
                    # 'val_estimation':val_images,
                    # 'val_gt':val_gts
                    },step=i)  
    else:
        wandb.log({'val_epe':val_aee,'lr': optim.param_groups[0]['lr']},step=i)
    if args.dataset == 'sintel': 
        for key in scene_epes:
            wandb.log({key:np.mean(scene_epes[key])},step=i)
    # if args is not None:
    #     alto_visu(flow,val_pred,args,i)
    model.train()
    return val_aee

def theta_prime(model,evaluation_indices,eval_dist):
    model.eval()
    val_epes = []
    val_images=[]
    val_gts=[]
    name="theta_prime"
    for s,t in enumerate(evaluation_indices):
        image1,image2,flow,valid = eval_dist[t] 
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        _, flow_pr = model(image1, image2, iters=12, test_mode=True)
        val_pred = padder.unpad(flow_pr[0]).cpu()
        # val_error, val_metrics = loss_fun(image1,image2,val_pred,flow,valid,args.gamma,model)
        # val_epes.append(val_metrics['epe'])
        epe = torch.sum((val_pred - flow)**2, dim=0).sqrt()
        epe = epe.view(-1)
        val = valid.view(-1) >= 0.5
        val_epes.append(epe[val].mean().item())
        val_images.append(wandb.Image(flow_img(val_pred)))
        val_gts.append(wandb.Image(flow_img(flow)))
    wandb_table=wandb.Table(columns=evaluation_indices.tolist(),data=[val_epes])
    # wandb.log({name:wandb_table,name+"_est":val_images,name+"_gts":val_gts})
    
    model.train()

    return np.mean(val_epes)

class PM_Validate:
    def __init__(self,args):
        self.ft_epe = []
        self.ft_loss = []
        self.val_epe=[]
        self.args = args
        return 

    def __call__(self,model,metrics,ft):
        val_result = evaluate.validate_single_kitti(model.module,iters=12,args=self.args)
        self.val_epe.append(val_result['kitti_epe'])
        self.ft_epe.append(metrics['epe'])
        self.ft_loss.append(metrics['loss_total'])
        if ft['iter'] %10 == 9 or ft['iter'] in np.arange(self.args.little_step):
            iters_image = wandb.Image(flow_img(ft['flow_pred'][0]))
            iters_gts = wandb.Image(flow_img(ft['flow']))
            iter_rgb = wandb.Image(flow_img(ft['image1'],rgb=True))
            wandb.log({"ft_after_meta_epe":np.mean(self.ft_epe),
                        "ft_after_meta_loss":np.mean(self.ft_loss),
                        "ft_after_meta_val":np.mean(self.val_epe),
                        # "ft_after_meta_flow":iters_image,
                        # "ft_after_meta_gt":iters_gts,
                        # "ft_after_meta_rgb":iter_rgb,
                        "real_step":ft['iter']},step=ft['iter']+1)
            self.ft_epe = []
            self.ft_loss = []
            self.val_epe = []
        return 
