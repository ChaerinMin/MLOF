# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import os
import os.path as osp
import random
import socket
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

from core.utils import frame_utils
from core.utils.augmentor import (FlowAugmentor, SparseFlowAugmentor,
                                  SparseFlowAugmentorRaw)

def sintel_samples(example_dataset,scenes,k):

    extra_info = example_dataset.extra_info
    dict = {}
    global_ids = []
    for id,info in enumerate(extra_info):
        dict[info[0]] = dict.get(info[0],[]) + [id]
    random.seed(90)
    for s in scenes:
        ids = dict[s]
        if len(ids)<k:
            raise ValueError("too much metaset data for a scene")
        global_ids.extend(random.sample(ids,k))
    print(global_ids)
    input()
    return global_ids

def cross_samples(dataset,cross_v,task_dist=None,args=None): 
    assert dataset == 'kitti' or dataset == 'sintel'
    adapt = []
    eval = []
    if dataset == 'kitti':
        eval=None
        if cross_v == 1: #Sm20_1
            adapt = list(np.arange(start=5,stop=200,step=10))
            eval = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 136, 137, 138, 139, 140, 141, 142, 143, 144, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159, 160, 161, 162, 163, 164, 166, 167, 168, 169, 170, 171, 172, 173, 174, 176, 177, 178, 179, 180, 181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199]
        elif cross_v == 2: #Sm20_2
            adapt = [84, 147, 28, 56, 108, 78, 99, 71, 150, 146, 76, 137, 43, 16, 62, 140, 193, 176, 4, 128]
            eval = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 77, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 103, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 141, 142, 143, 144, 145, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 194, 195, 196, 197, 198, 199]
        elif cross_v == 3: #Sm20_3
            adapt = [164, 183, 47, 143, 177, 110, 58, 149, 160, 46, 90, 30, 189, 87, 121, 63, 120, 153, 198, 126]
            eval = [0, 1, 2, 3, 4, 5, 6, 7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 26,  27,  28,  29,  31,  32,  33,  34,  35,  36,  37,  38,  39, 40,  41,  42,  43,  44,  45,  48,  49,  50,  51,  52,  53,  54, 55,  56,  57,  59,  60,  61,  62,  64,  65,  66,  67,  68,  69, 70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82, 83,  84,  85,  86,  88,  89,  91,  92,  93,  94,  95,  96,  97, 98,  99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 122, 123, 124, 125, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 151, 152, 154, 155, 156, 157, 158, 159, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 199]
        elif cross_v == 4: #Sm160_1
            adapt = [0, 1, 3, 5, 6, 7, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 24, 25, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 81, 82, 83, 84, 85, 86, 87, 90, 91, 92, 93, 94, 95, 97, 98, 99, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 120, 121, 122, 124, 125, 127, 128, 129, 130, 131, 132, 135, 136, 137, 138, 139, 141, 142, 143, 144, 145, 147, 148, 150, 151, 152, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 178, 179, 181, 182, 183, 185, 186, 187, 188, 189, 190, 191, 192, 194, 195, 196, 197, 198]
            eval = [ 17, 177, 159, 126, 106,  96, 140, 100,   2,  39, 133, 123, 134, 62, 38,  40,  80,  76,   8,  66,  79, 193,  44,  57, 117, 180,88,  23,   4,  30, 149,  16, 146, 184,   9, 119,  89,  26, 199,63]
        elif cross_v == 5: #Sm160_2
            adapt = [0, 2, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 81, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 104, 106, 107, 108, 110, 111, 112, 113, 114, 116, 117, 118, 120, 121, 122, 123, 124, 126, 127, 129, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 148, 149, 150, 151, 154, 155, 156, 157, 158, 159, 160, 162, 163, 165, 166, 167, 168, 169, 171, 172, 173, 175, 176, 177, 178, 181, 182, 183, 184, 186, 187, 189, 190, 191, 193, 194, 195, 196, 197, 198, 199]
            eval = [1, 3, 5, 11, 13, 20, 44, 50, 53, 54, 65, 70, 71, 80, 82, 83, 87, 101, 103, 105, 109, 115, 119, 125, 128, 130, 131, 143, 147, 152, 153, 161, 164, 170, 174, 179, 180, 185, 188, 192]
        elif cross_v == 6: #Sm160_3
            adapt = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 18, 19, 21, 22, 23, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 47, 49, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86, 87, 88, 89, 90, 92, 93, 95, 96, 97, 98, 99, 100, 102, 104, 106, 107, 108, 109, 112, 113, 114, 115, 116, 118, 119, 120, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 144, 145, 148, 150, 151, 152, 153, 155, 156, 158, 159, 160, 161, 162, 163, 165, 166, 168, 169, 171, 172, 173, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
            eval = [5, 12, 15, 16, 17, 20, 24, 29, 32, 39, 46, 48, 50, 51, 58, 64, 67, 82, 83, 91, 94, 101, 103, 105, 110, 111, 117, 121, 122, 140, 146, 147, 149, 154, 157, 164, 167, 170, 174, 178]
        elif cross_v == 7: #Sm10_1
            adapt = list(np.arange(start=0,stop=200,step=20)) #[0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
        elif cross_v == 8: #Sm5_1
            adapt = [0, 50, 100, 150, 199]
        elif cross_v == 9: #Sm100_1
            adapt = [0, 1, 4, 5, 7, 11, 14, 15, 16, 20, 21, 24, 25, 26, 27, 29, 30, 34, 36, 42, 45, 46, 47, 48, 49, 53, 55, 56, 57, 58, 59, 63, 65, 68, 69, 71, 72, 74, 77, 80, 81, 83, 87, 89, 90, 97, 100, 101, 102, 104, 105, 106, 107, 110, 111, 112, 114, 115, 116, 118, 120, 121, 126, 128, 129, 133, 137, 142, 143, 149, 151, 152, 153, 154, 156, 157, 158, 160, 161, 164, 166, 167, 168, 171, 172, 176, 177, 179, 182, 183, 189, 190, 191, 193, 194, 195, 196, 197, 198, 199]
        elif cross_v == 10: #Sm5_2
            adapt = [30,70,120,160,190]
        elif cross_v == 11: #Sm5_3
            adapt = [20,60,110,130,170]
        elif cross_v == 12: #Sm10_2
            adapt = [10,30,50,70,90,110,130,150,170,190] 
        elif cross_v==13: #Sm10_3
            adapt = [5,25,45,65,85,105,125,145,165,175]
        elif cross_v==14: #Sm50_1 
            adapt = [3, 5, 13, 15, 21, 22, 23, 25, 29, 35, 37, 44, 45, 47, 51, 54, 55, 56, 64, 65, 70, 73, 75, 76, 80, 85, 95, 101, 104, 110, 115, 118, 125, 126, 132, 135, 142, 145, 148, 152, 154, 155, 165, 169, 175, 183, 185, 189, 193, 195]
        elif cross_v==15: #Sm100_2
            adapt = [0, 2, 4, 6, 7, 10, 11, 14, 15, 17, 20, 21, 25, 26, 27, 28, 29, 30, 33, 35, 36, 38, 40, 41, 42, 43, 44, 48, 49, 52, 56, 62, 63, 65, 66, 71, 72, 74, 78, 82, 83, 90, 91, 93, 94, 95, 96, 97, 99, 100, 104, 106, 108, 109, 112, 113, 114, 115, 119, 120, 121, 123, 126, 127, 128, 130, 133, 135, 138, 144, 145, 149, 152, 153, 155, 157, 158, 163, 164, 165, 167, 168, 170, 171, 173, 174, 175, 178, 179, 180, 181, 186, 188, 189, 190, 195, 196, 197, 198, 199]
        elif cross_v==16: #Sm100_3
            adapt = [0, 2, 3, 5, 8, 11, 12, 13, 14, 16, 19, 23, 29, 32, 35, 36, 37, 38, 40, 41, 43, 44, 48, 49, 50, 51, 52, 53, 54, 60, 62, 63, 64, 68, 70, 71, 74, 77, 78, 79, 82, 84, 86, 87, 90, 92, 94, 96, 97, 98, 99, 104, 105, 106, 107, 108, 111, 113, 115, 117, 122, 124, 125, 127, 128, 129, 130, 134, 136, 137, 138, 140, 141, 144, 145, 147, 148, 152, 158, 161, 162, 163, 164, 169, 171, 173, 175, 176, 178, 179, 181, 183, 185, 186, 187, 193, 194, 195, 197, 198]
        elif cross_v==17: #Sm50_2
            adapt = [5, 6, 8, 13, 23, 40, 43, 44, 52, 53, 56, 58, 64, 68, 73, 81, 85, 86, 92, 97, 99, 102, 104, 105, 107, 113, 114, 117, 119, 121, 122, 124, 127, 128, 129, 131, 132, 133, 136, 137, 138, 141, 148, 149, 163, 169, 173, 180, 181, 186]
        else:
            raise ValueError
        if eval is None:
            eval = list(set(range(200))-set(adapt))
    elif dataset == 'sintel':
        if cross_v == 1:
            adapt = [611, 32, 658, 392, 366, 306, 13, 1029, 704, 952, 20, 994, 730, 942, 302, 127, 408, 463, 687, 507]
        elif cross_v == 2:
            adapt = [119, 68, 234, 946, 327, 1005, 65, 763, 406, 984, 10, 793, 390, 32, 451, 109, 469, 559, 409, 156]
        elif cross_v == 3:
            adapt = [351, 540, 238, 51, 891, 143, 65, 948, 396, 886, 602, 57, 505, 125, 795, 803, 318, 750, 984, 869]
        elif cross_v in [7,8,10]:
            adapt = sintel_samples(task_dist,args.sm_scenes,args.num_per_scene)
        elif cross_v == 9:
            scenes = ['bandage_1', 'bandage_2', 'bamboo_1', 'bamboo_2', 'market_2', 'market_6', 'alley_1', 'temple_3', 'cave_2', 'alley_2', 'ambush_5', 'mountain_1', 'market_5', 'temple_2', 'ambush_2', 'sleeping_2', 'sleeping_1', 'ambush_4', 'ambush_7', 'ambush_6', 'cave_4', 'shaman_2', 'shaman_3']
            adapt = sintel_samples(task_dist,scenes,args.num_per_scene)
        elif cross_v == 11:
            adapt = [3, 7, 13, 15, 19, 21, 41, 44, 68, 75, 77, 84, 111, 120, 123, 151, 163, 168, 169, 170, 173, 185, 199, 200, 225, 257, 284, 296, 311, 333, 336, 339, 343, 344, 371, 401, 409, 423, 425, 436, 448, 454, 482, 491, 492, 517, 523, 526, 528, 542, 544, 564, 566, 587, 590, 595, 597, 609, 617, 619, 637, 646, 649, 661, 666, 682, 683, 684, 693, 701, 703, 716, 721, 758, 759, 763, 791, 801, 802, 803, 826, 841, 843, 856, 868, 870, 891, 894, 897, 906, 915, 935, 939, 945, 971, 986, 1011, 1018, 1021, 1031]
        else:
            raise ValueError
        eval = list(set(range(1041))-set(adapt))
    else:
        raise NotImplementedError
    adapt, eval = np.array(adapt), np.array(eval)
    return adapt, eval

class FlowDatasetRaw(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentorRaw(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.occ_list = None
        self.seg_list = None
        self.seg_inv_list = None

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()

        if self.seg_list is not None:
            f_in = np.array(frame_utils.read_gen(self.seg_list[index]))
            seg_r = f_in[:, :, 0].astype('int32')
            seg_g = f_in[:, :, 1].astype('int32')
            seg_b = f_in[:, :, 2].astype('int32')
            seg_map = (seg_r * 256 + seg_g) * 256 + seg_b
            seg_map = torch.from_numpy(seg_map)

        if self.seg_inv_list is not None:
            seg_inv = frame_utils.read_gen(self.seg_inv_list[index])
            seg_inv = np.array(seg_inv).astype(np.uint8)
            seg_inv = torch.from_numpy(seg_inv // 255).bool()

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.occ_list is not None:
            return img1, img2, flow, valid.float(), occ, self.occ_list[index]
        elif self.seg_list is not None and self.seg_inv_list is not None:
            return img1, img2, flow, valid.float(), seg_map, seg_inv
        else:
            return img1, img2, flow, valid.float(), self.image_list[index][0] #, self.extra_info[index]

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.flow_list_noc=[]
        self.image_list = []
        self.extra_info = []
        self.occ_list = None
        self.seg_list = None
        self.seg_inv_list = None

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            flow_noc, valid_noc = frame_utils.readFlowKITTI(self.flow_list_noc[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])
            flow_noc = frame_utils.read_gen(self.flow_list_noc[index])

        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()

        if self.seg_list is not None:
            f_in = np.array(frame_utils.read_gen(self.seg_list[index]))
            seg_r = f_in[:, :, 0].astype('int32')
            seg_g = f_in[:, :, 1].astype('int32')
            seg_b = f_in[:, :, 2].astype('int32')
            seg_map = (seg_r * 256 + seg_g) * 256 + seg_b
            seg_map = torch.from_numpy(seg_map)

        if self.seg_inv_list is not None:
            seg_inv = frame_utils.read_gen(self.seg_inv_list[index])
            seg_inv = np.array(seg_inv).astype(np.uint8)
            seg_inv = torch.from_numpy(seg_inv // 255).bool()

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        flow_noc = np.array(flow_noc).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid,flow_noc = self.augmentor(img1, img2, flow, valid,flow_noc) 
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flow_noc = torch.from_numpy(flow_noc).permute(2, 0, 1).float()
        gt_occ = frame_utils.get_GTocc(flow,flow_noc)

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.occ_list is not None:
            return img1, img2, flow, valid.float(), occ, self.occ_list[index], gt_occ
        elif self.seg_list is not None and self.seg_inv_list is not None:
            return img1, img2, flow, valid.float(), seg_map, seg_inv, gt_occ
        else:
            return img1, img2, flow, valid.float(), gt_occ#, self.extra_info[index]


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDatasetRaw):
     def __init__(self, aug_params=None, split='training', root='./datasets/Sintel', dstype='clean',
                 occlusion=False, segmentation=False):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        # occ_root = osp.join(root, split, 'occlusions')
        # occ_root = osp.join(root, split, 'occ_plus_out')
        # occ_root = osp.join(root, split, 'in_frame_occ')
        occ_root = osp.join(root, split, 'out_of_frame')

        seg_root = osp.join(root, split, 'segmentation')
        seg_inv_root = osp.join(root, split, 'segmentation_invalid')
        self.segmentation = segmentation
        self.occlusion = occlusion
        if self.occlusion:
            self.occ_list = []
        if self.segmentation:
            self.seg_list = []
            self.seg_inv_list = []

        if split == 'test':
            self.is_test = True

        dirs = sorted(os.listdir(image_root))
        for scene in dirs:
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))
                if self.occlusion:
                    self.occ_list += sorted(glob(osp.join(occ_root, scene, '*.png')))
                if self.segmentation:
                    self.seg_list += sorted(glob(osp.join(seg_root, scene, '*.png')))
                    self.seg_inv_list += sorted(glob(osp.join(seg_inv_root, scene, '*.png')))
        
        return 


class SintelSelect(MpiSintel):
    def __init__(self, id_list,aug_params=None, split='training', root='./datasets/Sintel', dstype='clean',
                 occlusion=False, segmentation=False):
        super().__init__(aug_params,split=split,root=root,dstype=dstype,occlusion=occlusion,segmentation=segmentation)
        if occlusion or segmentation:
            raise NotImplementedError
        assert max(id_list) < len(self.image_list)
        self.image_list = [self.image_list[i] for i in id_list]
        self.extra_info = [self.extra_info[i] for i in id_list]
        self.flow_list = [self.flow_list[i] for i in id_list]




class FlyingChairs(FlowDatasetRaw):
    def __init__(self, aug_params=None, split='training', root='./datasets/FlyingChairs/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('./datasets/FlyingChairs/FlyingChairs_train_val.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='./datasets/FlyingThings3D', split='training', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        if split == 'training':
            for cam in ['left']:
                for direction in ['into_future', 'into_past']:
                    image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, '*.png')) )
                        flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                        for i in range(len(flows)-1):
                            if direction == 'into_future':
                                self.image_list += [ [images[i], images[i+1]] ]
                                self.flow_list += [ flows[i] ]
                            elif direction == 'into_past':
                                self.image_list += [ [images[i+1], images[i]] ]
                                self.flow_list += [ flows[i+1] ]

        elif split == 'validation':
            for cam in ['left']:
                for direction in ['into_future', 'into_past']:
                    image_dirs = sorted(glob(osp.join(root, dstype, 'TEST/*/*')))
                    image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                    flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TEST/*/*')))
                    flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                    for idir, fdir in zip(image_dirs, flow_dirs):
                        images = sorted(glob(osp.join(idir, '*.png')))
                        flows = sorted(glob(osp.join(fdir, '*.pfm')))
                        for i in range(len(flows) - 1):
                            if direction == 'into_future':
                                self.image_list += [[images[i], images[i + 1]]]
                                self.flow_list += [flows[i]]
                            elif direction == 'into_past':
                                self.image_list += [[images[i + 1], images[i]]]
                                self.flow_list += [flows[i + 1]]

                valid_list = np.loadtxt('things_val_test_set.txt', dtype=np.int32)
                self.image_list = [self.image_list[ind] for ind, sel in enumerate(valid_list) if sel]
                self.flow_list = [self.flow_list[ind] for ind, sel in enumerate(valid_list) if sel]
      
class KITTIRaw(FlowDatasetRaw):
    def __init__(self, aug_params=None, split='training', root='./datasets/KITTI'):
        super(KITTIRaw, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='./datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            self.flow_list_noc = sorted(glob(osp.join(root, 'flow_noc/*_10.png')))

class KITTISelect(FlowDataset):
    def __init__(self, id_list,aug_params=None, split='training', root='./datasets/KITTI'):
        super(KITTISelect, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = [osp.join(root, 'image_2/{}_10.png'.format(str(img_id).rjust(6,'0'))) for img_id in id_list]
        images2 = [osp.join(root, 'image_2/{}_11.png'.format(str(img_id).rjust(6,'0'))) for img_id in id_list]

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = [osp.join(root, 'flow_occ/{}_10.png'.format(str(img_id).rjust(6,'0'))) for img_id in id_list]
            self.flow_list_noc = [osp.join(root, 'flow_noc/{}_10.png'.format(str(img_id).rjust(6,'0'))) for img_id in id_list]


class SingleKITTI(FlowDataset):
    def __init__(self, img_id,aug_params=None, split='training', root='./datasets/KITTI'):
        super(SingleKITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = [osp.join(root, 'image_2/{}_10.png'.format(str(img_id).rjust(6,'0')))]
        images2 = [osp.join(root, 'image_2/{}_11.png'.format(str(img_id).rjust(6,'0')))]

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = [osp.join(root, 'flow_occ/{}_10.png'.format(str(img_id).rjust(6,'0')))]
            self.flow_list_noc = [osp.join(root, 'flow_noc/{}_10.png'.format(str(img_id).rjust(6,'0')))]

class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='./datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding training set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass', split='training')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass', split='training')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H': 
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            # if args.id_list:
            #     kitti=KITTISelect(args.id_list,{'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            #     train_dataset = 100*sintel_clean + 100*sintel_final + len(args.id_list)*kitti + 5*hd1k + things
            # else:
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')


    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

def train_test_loader(args,test_range=None):
    if args.dataset == 'CSK':
        chairs_params = {'crop_size': [368,496], 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        sintel_params = {'crop_size': [368,768], 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        chairs = FlyingChairs(chairs_params, split='training')
        sintel_clean = MpiSintel(sintel_params, split='training', dstype='clean')
        sintel_final = MpiSintel(sintel_params, split='training', dstype='final')        
        kitti = KITTIRaw({'crop_size': [288,960], 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})

        test_dataset,trainset = val_extractor([sintel_clean,sintel_final,kitti,chairs],30)
    
        weights = sample_indices(trainset)
        train_dataset = trainset[0]
        for i in trainset[1:]:
            train_dataset += i
    train_sampler = data.WeightedRandomSampler(weights,args.batch_size)
    train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size,sampler=train_sampler,pin_memory=True,num_workers=8,drop_last=True)
    test_loader = data.DataLoader(test_dataset,batch_size=1,pin_memory=True,shuffle=False,num_workers=8,drop_last=False)

    return train_loader, test_loader 

def sample_indices(subsets):
    weights = []
    for subs in subsets:
        num = len(subs)
        if isinstance(subs.dataset,MpiSintel):
            indices = num*[1.0/num/2]
        else:
            indices = num*[1.0/num]
        weights.extend(indices)
    return weights

def val_extractor(datasets,val_num):
    val_sets=[]
    train_sets = []
    assert val_num%2 ==0
    assert isinstance(datasets,list)
    for dset in datasets:
        if isinstance(dset,MpiSintel):
            valset, trainset = data.random_split(dset,[val_num//2,len(dset)-val_num//2], generator=torch.Generator().manual_seed(42))
        else:

            valset, trainset = data.random_split(dset,[val_num,len(dset)-val_num], generator=torch.Generator().manual_seed(52))
            # trainset = dset - valset
        val_sets.append(valset)
        train_sets.append(trainset)
    vals = val_sets[0]
    for i in val_sets[1:]:
        vals += i
    return vals, train_sets 

