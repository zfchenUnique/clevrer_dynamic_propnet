import os
import torch
import time
import random
import numpy as np
import gzip
import pickle
import h5py
import json

import pycocotools._mask as _mask
import cv2
from skimage import io, transform
from PIL import Image

import multiprocessing as mp
import scipy.spatial as spatial
from sklearn.cluster import MiniBatchKMeans

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from utils import prepare_relations, convert_mask_to_bbox, crop, encode_attr
from utils import normalize, check_attr, get_identifier, get_identifiers
from utils import check_same_identifier, check_same_identifiers, check_contain_id
from utils import get_masks, check_valid_masks, check_duplicate_identifier
from utils import rand_float, init_stat, combine_stat, load_data, store_data
from utils import decode, make_video

import utils_tube as utilsTube
from utils_tube import check_box_in_tubes 
import pdb
import pycocotools.mask as cocoMask
import copy
import jactorch.transforms.bbox as T


def decode_mask_to_box(mask, crop_box_size, H, W):
    bbx_xywh_ori = cocoMask.toBbox(mask)
    bbx_xywh = copy.deepcopy(bbx_xywh_ori)
    bbx_xyxy = copy.deepcopy(bbx_xywh)
    crop_box = copy.deepcopy(bbx_xywh)
    
    bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
    bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
    
    bbx_xywh[0] = bbx_xywh[0]*1.0/mask['size'][1] 
    bbx_xywh[2] = bbx_xywh[2]*1.0/mask['size'][1] 
    bbx_xywh[1] = bbx_xywh[1]*1.0/mask['size'][0] 
    bbx_xywh[3] = bbx_xywh[3]*1.0/mask['size'][0] 
    bbx_xywh[0] = bbx_xywh[0] + bbx_xywh[2]/2.0 
    bbx_xywh[1] = bbx_xywh[1] + bbx_xywh[3]/2.0 

    crop_box[1] = int((bbx_xyxy[0])*W/mask['size'][1]) # w
    crop_box[0] = int((bbx_xyxy[1])*H/mask['size'][0]) # h
    crop_box[2] = int(crop_box_size[0])
    crop_box[3] = int(crop_box_size[1])

    crop_box_v2 = copy.deepcopy(crop_box)
    off_set_x = max(int(0.5*(crop_box_size[0]-bbx_xywh_ori[2]*W/mask['size'][1])), 0)
    off_set_y = max(int(0.5*(crop_box_size[1]-bbx_xywh_ori[3]*H/mask['size'][0])), 0)
    crop_box_v2[0] = crop_box_v2[0] - off_set_y # w
    crop_box_v2[1] = crop_box_v2[1] - off_set_x # h
    #pdb.set_trace()


    ret = np.ones((4, crop_box_size[0], crop_box_size[1]))
    ret[0, :, :] *= bbx_xywh[0]
    ret[1, :, :] *= bbx_xywh[1]
    ret[2, :, :] *= bbx_xywh[2]
    ret[3, :, :] *= bbx_xywh[3]
    ret = torch.FloatTensor(ret)
    return bbx_xyxy, ret, crop_box.astype(int), crop_box_v2.astype(int)   
    


def collate_fn(data):
    return data[0]


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class PhysicsCLEVRDataset(Dataset):

    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.loader = default_loader
        self.data_dir = args.data_dir
        self.label_dir = args.label_dir
        self.prp_dir = args.prp_dir
        self.ann_dir = args.ann_dir
        self.tube_dir = args.tube_dir 

        self.valid_idx_lst = 'valid_idx_' + self.phase + '.txt'
        self.H = 100
        self.W = 150
        self.bbox_size = 24

        ratio = self.args.train_valid_ratio
        n_train = round(self.args.n_rollout * ratio)
        if phase == 'train':
            self.st_idx = 0
            self.n_rollout = n_train
        elif phase == 'valid':
            self.st_idx = n_train
            self.n_rollout = self.args.n_rollout - n_train
        else:
            raise AssertionError("Unknown phase")

        if self.args.gen_valid_idx:
            if self.args.version=='v3':
                self.gen_valid_idx_from_tube_info_v3()
            else:
                self.gen_valid_idx_from_tube_info()
        else:
            self.read_valid_idx()


        self.img_transform = T.Compose([
                T.Resize(self.args.img_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.W = 480; self.H = 320

    def read_valid_idx(self):
        # if self.phase == 'train':
        # return
        print("Reading valid idx ...")
        self.n_valid_idx = 0
        self.valid_idx = []
        self.metadata = []
        fin = open(self.valid_idx_lst, 'r').readlines()

        self.n_valid_idx = len(fin)
        for i in range(self.n_valid_idx):
            a = int(fin[i].strip().split(' ')[0])
            b = int(fin[i].strip().split(' ')[1])
            self.valid_idx.append((a, b))

        for i in range(self.st_idx, self.st_idx + self.n_rollout):
            if i % 500 == 0:
                print("Reading valid idx %d/%d" % (i, self.st_idx + self.n_rollout))

            vid = int(i/1000)
            ann_full_dir = os.path.join(self.ann_dir, 'annotation_%02d000-%02d000'%(vid, vid+1))
            #pk_path = os.path.join(self.tube_dir, 'annotation_%05d.pk' % i)
            pk_path = os.path.join(self.tube_dir, 'proposal_%05d.pk' % i)
            prp_path = os.path.join(self.prp_dir, 'proposal_%05d.json' % i)
            ann_path = os.path.join(ann_full_dir, 'annotation_%05d.json' % i)

            if not os.path.isfile(pk_path):
                pk_path = os.path.join(self.tube_dir, 'annotation_%05d.pk' % i)

            tubes_info = utilsTube.pickleload(pk_path)
            prp_info = utilsTube.jsonload(prp_path)
            data = utilsTube.jsonload(ann_path)
            data['tubes'] = tubes_info['tubes']
            data['proposals'] = prp_info 
            self.metadata.append(data)


    def gen_valid_idx_from_tube_info_v3(self):
        print("Preprocessing valid idx ...")
        self.n_valid_idx = 0
        self.valid_idx = []
        self.metadata = []
        fout = open(self.valid_idx_lst, 'w')

        n_his = self.args.n_his
        frame_offset = self.args.frame_offset

        for i in range(self.st_idx, self.st_idx + self.n_rollout):
            if i % 500 == 0:
                print("Preprocessing valid idx %d/%d" % (i, self.st_idx + self.n_rollout))

            vid = int(i/1000)
            ann_full_dir = os.path.join(self.ann_dir, 'annotation_%02d000-%02d000'%(vid, vid+1))
            #with open(os.path.join(self.label_dir, 'proposal_%05d.json' % i)) as f:
            #pk_path = os.path.join(self.tube_dir, 'annotation_%05d.pk' % i)
            pk_path = os.path.join(self.tube_dir, 'proposal_%05d.pk' % i)
            prp_path = os.path.join(self.prp_dir, 'proposal_%05d.json' % i)
            ann_path = os.path.join(ann_full_dir, 'annotation_%05d.json' % i)
        
            if not os.path.isfile(pk_path):
                pk_path = os.path.join(self.tube_dir, 'annotation_%05d.pk' % i)

            tubes_info = utilsTube.pickleload(pk_path)
            prp_info = utilsTube.jsonload(prp_path)
            data = utilsTube.jsonload(ann_path)
            data['tubes'] = tubes_info['tubes']
            data['proposals'] = prp_info 
            self.metadata.append(data)
            
            #pdb.set_trace()
            for j in range(
                n_his * frame_offset,
                len(data['proposals']['frames']) - frame_offset):

                frm_list = []
                objects = data['proposals']['frames'][j]['objects']
                frm_list.append(j)
                n_object_cur = len(objects)
                valid = True


                if not check_box_in_tubes(objects, j, data['tubes']):
                    valid = False

                # check whether history window is valid
                for k in range(n_his):
                    idx = j - (k + 1) * frame_offset
                    objects = data['proposals']['frames'][idx]['objects']
                    frm_list.append(idx)
                    n_object = len(objects)

                    if (not valid) or n_object != n_object_cur:
                        valid = False
                        break
                
                    if not check_box_in_tubes(objects, idx, data['tubes']):
                        valid = False
                    
                    # check valid tube, making box valid
                    if k==(n_his-1):
                        tube_num = len(data['tubes'])
                        for tube_id in range(tube_num):
                            tmp_box = data['tubes'][tube_id][idx]
                            if tmp_box == [0, 0, 1, 1]:
                                valid = False 
                                break 

                if valid:
                    # check whether the target is valid
                    idx = j + frame_offset
                    objects_nxt = data['proposals']['frames'][idx]['objects']
                    n_object_nxt = len(objects_nxt)
                    frm_list.append(idx)

                    if (not valid) or n_object_nxt != n_object_cur:
                        valid = False


                    if utilsTube.check_object_inconsistent_identifier(frm_list, data['tubes']):
                        valid = False

                    if utilsTube.checking_duplicate_box_among_tubes(frm_list, data['tubes']):
                        valid = False
                    
                    if not check_box_in_tubes(objects_nxt, idx, data['tubes']):
                        valid = False



                if valid:
                    self.valid_idx.append((i - self.st_idx, j))
                    fout.write('%d %d\n' % (i - self.st_idx, j))
                    self.n_valid_idx += 1

        fout.close()

    def gen_valid_idx_from_tube_info(self):
        print("Preprocessing valid idx ...")
        self.n_valid_idx = 0
        self.valid_idx = []
        self.metadata = []
        fout = open(self.valid_idx_lst, 'w')

        n_his = self.args.n_his
        frame_offset = self.args.frame_offset

        for i in range(self.st_idx, self.st_idx + self.n_rollout):
            if i % 500 == 0:
                print("Preprocessing valid idx %d/%d" % (i, self.st_idx + self.n_rollout))

            vid = int(i/1000)
            ann_full_dir = os.path.join(self.ann_dir, 'annotation_%02d000-%02d000'%(vid, vid+1))
            #with open(os.path.join(self.label_dir, 'proposal_%05d.json' % i)) as f:
            #pk_path = os.path.join(self.tube_dir, 'annotation_%05d.pk' % i)
            pk_path = os.path.join(self.tube_dir, 'proposal_%05d.pk' % i)
            prp_path = os.path.join(self.prp_dir, 'proposal_%05d.json' % i)
            ann_path = os.path.join(ann_full_dir, 'annotation_%05d.json' % i)
        
            if not os.path.isfile(pk_path):
                pk_path = os.path.join(self.tube_dir, 'annotation_%05d.pk' % i)

            tubes_info = utilsTube.pickleload(pk_path)
            prp_info = utilsTube.jsonload(prp_path)
            data = utilsTube.jsonload(ann_path)
            data['tubes'] = tubes_info['tubes']
            data['proposals'] = prp_info 
            self.metadata.append(data)
            
            #pdb.set_trace()
            for j in range(
                n_his * frame_offset,
                len(data['proposals']['frames']) - frame_offset):

                frm_list = []
                objects = data['proposals']['frames'][j]['objects']
                frm_list.append(j)
                n_object_cur = len(objects)
                valid = True


                if not check_box_in_tubes(objects, j, data['tubes']):
                    valid = False

                # check whether history window is valid
                for k in range(n_his):
                    idx = j - (k + 1) * frame_offset
                    objects = data['proposals']['frames'][idx]['objects']
                    frm_list.append(idx)
                    n_object = len(objects)

                    if (not valid) or n_object != n_object_cur:
                        valid = False
                        break
                
                    if not check_box_in_tubes(objects, idx, data['tubes']):
                        valid = False

                if valid:
                    # check whether the target is valid
                    idx = j + frame_offset
                    objects_nxt = data['proposals']['frames'][idx]['objects']
                    n_object_nxt = len(objects_nxt)
                    frm_list.append(idx)

                    if (not valid) or n_object_nxt != n_object_cur:
                        valid = False


                    if utilsTube.check_object_inconsistent_identifier(frm_list, data['tubes']):
                        valid = False

                    if utilsTube.checking_duplicate_box_among_tubes(frm_list, data['tubes']):
                        valid = False
                    
                    if not check_box_in_tubes(objects_nxt, idx, data['tubes']):
                        valid = False

                if valid:
                    self.valid_idx.append((i - self.st_idx, j))
                    fout.write('%d %d\n' % (i - self.st_idx, j))
                    self.n_valid_idx += 1

        fout.close()

    '''
    def read_valid_idx(self):
        fin = open(self.valid_idx_lst, 'r').readlines()
        self.n_valid_idx = len(fin)
        self.valid_idx = []
        for i in range(len(fin)):
            idx = [int(x) for x in fin[i].strip().split(' ')]
            self.valid_idx.append((idx[0], idx[1]))
    '''

    def __len__(self):
        return self.n_valid_idx


    def __getitem__(self, idx):
        if self.args.data_ver =='v1':
            return self.__getitem__v1(idx)
        elif self.args.data_ver =='v2':
            return self.__getitem__v2(idx)
        elif self.args.data_ver =='v3':
            return self.__getitem__v2(idx)

    def __getitem__v2(self, idx):
        n_his = self.args.n_his
        frame_offset = self.args.frame_offset
        idx_video, idx_frame = self.valid_idx[idx][0], self.valid_idx[idx][1]

        objs = []
        attrs = []
        img_list = [] 
        obj_num = len(self.metadata[idx_video]['tubes'])
        smp_tube_info = {obj_id:{'boxes': [], 'frm_name': []} for obj_id in range(obj_num)}
        frm_idx_list  = []
        box_seq = {obj_id: [] for obj_id in range(obj_num)}
        invalid_tube_id_list = []

        for i in range(
            idx_frame - n_his * frame_offset,
            idx_frame + frame_offset + 1, frame_offset):

            frame = self.metadata[idx_video]['proposals']['frames'][i]
            #frame_filename = frame['frame_filename']
            frame_filename = os.path.join('video_'+str(idx_video).zfill(5), str(frame['frame_index']+1)+'.png') 
            vid = int(idx_video/1000)
            ann_full_dir = os.path.join(self.data_dir, 'image_%02d000-%02d000'%(vid, vid+1))
            img_full_path = os.path.join(ann_full_dir, frame_filename)
            img = Image.open(img_full_path).convert('RGB')
            W_ori, H_ori = img.size
            img, _ = self.img_transform(img, np.array([0, 0, 1, 1]))
            img_list.append(img)
            frm_idx_list.append(i)

            img_size = self.args.img_size
            ratio = img_size / min(H_ori, W_ori)
            ### prepare object inputs
            object_inputs = []
            for j in range(obj_num):
                bbox_xyxy = self.metadata[idx_video]['tubes'][j][i] 
                if bbox_xyxy == [0, 0, 1, 1]:
                    #invalid_tube_id_list.append(j)
                    #continue
                    box_seq[j].append(torch.tensor([-1, -1, -1, -1]).float())
                    
                else:
                    box_tensor_ori = torch.tensor(bbox_xyxy).float()
                    box_tensor_norm = box_tensor_ori.clone()
                    box_tensor_target = box_tensor_ori.clone()
                    box_tensor_target = box_tensor_target*ratio

                    box_tensor_norm[0] = box_tensor_norm[0]/W_ori
                    box_tensor_norm[2] = box_tensor_norm[2]/W_ori
                    box_tensor_norm[1] = box_tensor_norm[1]/H_ori
                    box_tensor_norm[3] = box_tensor_norm[3]/H_ori
                    box_xyhw = box_tensor_norm.clone()
                    box_xyhw[2] = box_xyhw[2] - box_xyhw[0]
                    box_xyhw[3] = box_xyhw[3] - box_xyhw[1]
                    box_xyhw[1] = box_xyhw[1] + box_xyhw[3]*0.5
                    box_xyhw[0] = box_xyhw[0] + box_xyhw[2]*0.5
                    
                    smp_tube_info[j]['boxes'].append(box_tensor_target)
                    smp_tube_info[j]['frm_name'].append(i)
                    box_seq[j].append(box_xyhw)

        smp_tube_info['box_seq'] = box_seq 
        smp_tube_info['frm_list'] = frm_idx_list
        img_tensor = torch.stack(img_list, 0)
        data = {}
        data['img_future'] = img_tensor 
        data['predictions'] = smp_tube_info 

        return data 


    def __getitem__v1(self, idx):
        n_his = self.args.n_his
        frame_offset = self.args.frame_offset
        idx_video, idx_frame = self.valid_idx[idx][0], self.valid_idx[idx][1]

        objs = []
        attrs = []
        img_list = [] 
        obj_num = len(self.metadata[idx_video]['tubes'])
        smp_tube_info = {obj_id:{'boxes': [], 'frm_name': []} for obj_id in range(obj_num)}
        frm_idx_list  = []
        box_seq = {obj_id: [] for obj_id in range(obj_num)}
        invalid_tube_id_list = []

        for i in range(
            idx_frame - n_his * frame_offset,
            idx_frame + frame_offset + 1, frame_offset):

            frame = self.metadata[idx_video]['proposals']['frames'][i]
            #frame_filename = frame['frame_filename']
            frame_filename = os.path.join('video_'+str(idx_video).zfill(5), str(frame['frame_index']+1)+'.png') 
            vid = int(idx_video/1000)
            ann_full_dir = os.path.join(self.data_dir, 'image_%02d000-%02d000'%(vid, vid+1))
            img_full_path = os.path.join(ann_full_dir, frame_filename)
            img = Image.open(img_full_path).convert('RGB')
            W_ori, H_ori = img.size
            img, _ = self.img_transform(img, np.array([0, 0, 1, 1]))
            img_list.append(img)
            frm_idx_list.append(i)

            img_size = self.args.img_size
            ratio = img_size / min(H_ori, W_ori)
            ### prepare object inputs
            object_inputs = []
            for j in range(obj_num):
                bbox_xyxy = self.metadata[idx_video]['tubes'][j][i] 
                if bbox_xyxy == [0, 0, 1, 1]:
                    invalid_tube_id_list.append(j)
                    continue 
                box_tensor_ori = torch.tensor(bbox_xyxy).float()
                box_tensor_norm = box_tensor_ori.clone()
                box_tensor_target = box_tensor_ori.clone()
                box_tensor_target = box_tensor_target*ratio

                box_tensor_norm[0] = box_tensor_norm[0]/W_ori
                box_tensor_norm[2] = box_tensor_norm[2]/W_ori
                box_tensor_norm[1] = box_tensor_norm[1]/H_ori
                box_tensor_norm[3] = box_tensor_norm[3]/H_ori
                box_xyhw = box_tensor_norm.clone()
                box_xyhw[2] = box_xyhw[2] - box_xyhw[0]
                box_xyhw[3] = box_xyhw[3] - box_xyhw[1]
                box_xyhw[1] = box_xyhw[1] + box_xyhw[3]*0.5
                box_xyhw[0] = box_xyhw[0] + box_xyhw[2]*0.5
                
                smp_tube_info[j]['boxes'].append(box_tensor_target)
                smp_tube_info[j]['frm_name'].append(i)
                box_seq[j].append(box_xyhw)

        invalid_tube_id_list_unqiue = list(set(invalid_tube_id_list))
        for tube_id in sorted(invalid_tube_id_list_unqiue, reverse=True):
            del box_seq[tube_id]
            del smp_tube_info[tube_id]
        new_tube_idx = 0
        valid_tube_id_list = [tube_id for tube_id in range(obj_num) if tube_id not in invalid_tube_id_list_unqiue]
        new_smp_tube_info = {}
        new_box_seq = {}
        for new_idx, tube_id in enumerate(sorted(valid_tube_id_list, reverse=False)): 
            new_smp_tube_info[new_idx] = smp_tube_info[tube_id]
            new_box_seq[new_idx] = box_seq[tube_id]
        # TODO: solve it more elegantly
        if len(valid_tube_id_list)==0:
            return self.__getitem__(idx+1)

        new_smp_tube_info['box_seq'] = new_box_seq 
        new_smp_tube_info['frm_list'] = frm_idx_list
        img_tensor = torch.stack(img_list, 0)
        data = {}
        data['img_future'] = img_tensor 
        data['predictions'] = new_smp_tube_info 

        return data 

