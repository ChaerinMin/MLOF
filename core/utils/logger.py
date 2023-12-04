import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import wandb

from core.utils import flow_viz

def flow_img(flow,rgb=False,clip_flow=None):
    if flow.dim() == 4:
        flow = flow[-1]
    flow = flow.permute([1,2,0]).detach().cpu().numpy()
    assert flow.ndim == 3, 'input flow must have three dimensions'
    assert flow.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if rgb:
        flow_img=flow
    else:
        if clip_flow is not None:
            flow_img = flow_viz.flow_to_image(flow,clip_flow = clip_flow)
        else:
            flow_img = flow_viz.flow_to_image(flow)

    return flow_img

def error_map(pred,gt):
    if pred.dim() == 4:
        pred = pred[-1]
    pred = pred.permute([1,2,0]).detach().cpu().numpy()
    if gt.dim() == 4:
        gt = gt[0]
    gt = gt.permute([1,2,0]).detach().cpu().numpy()
    error_img,er_clip = flow_viz.plot_error(pred,gt)
    return error_img,er_clip

class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0 

        self.running_loss_dict = {}
        self.train_epe_list = []
        self.images = []
        self.gts = []
        self.rgb=[]
        self.errors=[]
        self.train_steps_list = []
        self.range_occ = []
        self.range_noc = []

        self.val_steps_list = []
        self.val_results_dict = {} 
        self.val_image_epe = []
        self.val_image_f1 = []
        self.val_images =[]
        self.val_gt_images =[]
        self.val_rgb=[]
        self.val_err = []
        self.vall_err_clip = []

        self.pred_flow = None
        self.gt_flow = None
        self.flow_write = ''

        if args.one_image>-1:
            # wandb_config ={
            #     "dataset" : args.dataset,
            #     "image_size" : args.image_size,
            #     "loss_fn" : args.loss_fn,
            #     "lr" : args.lr,
            #     "model" : args.model,
            #     "name" : args.name,
            #     "one_image" : args.one_image,
            #     "wkeeper" : args.wkeeper
            # }
            wandb.init(config=args,name="id"+str(args.one_image)+"_"+str(args.name),group='single_img',dir='/tmp')
        else:
            wandb.init(config=args,name=args.name,dir='/tmp')
        # wandb.watch(model)

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[steps {:6d}, lr {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        metrics_name = [k for k in sorted(self.running_loss_dict.keys())]

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        # print the training status
        # if self.total_steps + 1 == self.args.print_freq:
            # name_str = ("{}, "*len(metrics_name)).format(*metrics_name)
            # print(name_str)
        # print(training_str + metrics_str + time_left_hms)

        #wandb
        for data,name in zip(metrics_data,metrics_name):
            wandb.log({name : data},step=self.total_steps) 


        # logging running loss to total loss
        self.train_epe_list.append(np.mean(self.running_loss_dict['epe']))
        # self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []
        

    def push(self, metrics,images,gts,rgb,errors,er_clip,range_occ,range_noc):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []

            self.running_loss_dict[key].append(metrics[key])
        self.images = images
        self.gts = gts
        self.rgb = rgb
        self.errors= errors
        self.er_clip=er_clip
        self.range_occ = range_occ
        self.range_noc = range_noc

        if self.total_steps % self.args.print_freq ==0 or self.total_steps == 1:
            self._print_training_status()
            self.running_loss_dict = {}
            self.images=[]
            self.gts=[]
            self.rbg=[]
            self.errors=[]
            self.er_clip=[]
            self.range_occ=[]
            self.range_noc=[]

    def write_flow(self,pred,gt,val_id): 
        if self.flow_write:
            if pred.dim() == 4:
                pred = torch.squeeze(pred)
                gt = torch.squeeze(gt)

            pred = pred.permute([1,2,0]).detach().cpu().numpy()
            gt = gt.permute([1,2,0]).detach().cpu().numpy()

            flow_img = flow_viz.flow_to_image(pred)
            image = Image.fromarray(flow_img)
            gt_img = flow_viz.flow_to_image(gt)
            gt_image = Image.fromarray(gt_img)
            err_img,er_clip=flow_viz.plot_error(pred,gt) #h,w,2

            image.save(os.path.join(self.args.output,f'{val_id}{self.flow_write}_pred.png'))
            if self.flow_write == 'initial':
                gt_image.save(os.path.join(self.args.output,f'{val_id}{self.flow_write}_gt.png'))
        
        return
    

    def if_write(self,write):
        self.flow_write = write    

def write_diff(diff,diff_clip,args):
    diff_image = Image.fromarray(diff)
    diff_clip_image = Image.fromarray(diff_clip)
    diff_image.save(os.path.join(args.output,f'{args.one_image}_diff.png'))
    diff_clip_image.save(os.path.join(args.output,f'{args.one_image}_diff_clip20.png'))
    return 

class SelectionLogger:
    def __init__(self,args):
        self.name_txt = os.path.join(args.output,'Snet_out.txt')
        self.ex_txt = os.path.join(args.output,'ex_inner_epe.txt')
        self.topk_txt = os.path.join(args.output,'topk.txt')
        self.selected = None
        self.outvec = None

        return 

    def add(self,selected):
        self.selected = selected 

    def full_add(self,output):
        self.outvec = output

    def _print_screen(self):
        lang = f"outer {self.outer}, inner {self.inner} : topk "
        topk = self.selected
        print(lang,sep='')
        if topk is not None:
            for i in topk:
                print(i.item(),sep=' ')
            with open(self.topk_txt,'a') as f:
                if topk.dim() ==1 and topk.size(dim=0) == 3:
                    f.write(lang+f"  {topk[0].item()} {topk[1].item()} {topk[2].item()} \n")
                else:
                    raise NotImplementedError
            self.selected = None
        else:
            raise ValueError 

        return 
    
    def _print_fulltxt(self):
        if self.outvec is not None:
            lang = f"outer {self.outer}, inner {self.inner} :"
            with open(self.name_txt,'a') as f:
                f.write(lang+'\n')
                for num in torch.squeeze(self.outvec):
                    f.write(f'{num:.4f} ')
                f.write("\n\n")
            self.outvec = None
        else:
            raise ValueError 

        return 
    
    def trace_Sinput(self,imgid,epe):
        wandb.log({f"ex_outer_epe/metaimg_{imgid}" : epe})
        return 

    def printout(self,counts):
        self.outer = counts['outer']
        self.inner = counts['inner']
        self._print_screen()
        self._print_fulltxt()
    
    def print_Seffect(self,result,counts):
        info = f"outer {counts['outer']}, inner {counts['inner']}, training {counts['ex_epoch']} : epe "
        with open(self.ex_txt,'a') as f:
            f.write(info+f"{result:.4f}\n")
        return


def alto_visu(flow,val_pred,args,iters,name='',gt_save=True):
    if isinstance(val_pred,list):
        val_pred = val_pred[-1]
    if flow.dim() == 4:
        flow = flow[-1]
        val_pred = val_pred[-1]
    if name:
        name = name+"_"
    common_clip = flow_viz.determine_clip(flow)
    iters_image = Image.fromarray(flow_img(val_pred,clip_flow=common_clip))
    gt_image = Image.fromarray(flow_img(flow,clip_flow=common_clip))
    iters_image.save(os.path.join(args.output,f"{name}{iters+1}.png"))
    if gt_save:
        gt_image.save(os.path.join(args.output,f"{name}gt{iters+1}.png"))
    return 

    