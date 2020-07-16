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
            self.gen_valid_idx_from_tube_info()
        else:
            self.read_valid_idx()

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
        #pdb.set_trace()
        n_his = self.args.n_his
        frame_offset = self.args.frame_offset
        idx_video, idx_frame = self.valid_idx[idx][0], self.valid_idx[idx][1]

        objs = []
        attrs = []
        for i in range(
            idx_frame - n_his * frame_offset,
            idx_frame + frame_offset + 1, frame_offset):

            frame = self.metadata[idx_video]['proposals']['frames'][i]
            #frame_filename = frame['frame_filename']
            frame_filename = os.path.join('video_'+str(idx_video).zfill(5), str(frame['frame_index']+1)+'.png') 

            objects = frame['objects']
            n_objects = len(objects)

            vid = int(idx_video/1000)
            ann_full_dir = os.path.join(self.data_dir, 'image_%02d000-%02d000'%(vid, vid+1))
            img = self.loader(os.path.join(ann_full_dir, frame_filename))
            img = np.array(img)[:, :, ::-1].copy()
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float) / 255.

            ### prepare object inputs
            object_inputs = []
            for j in range(n_objects):
                material = objects[j]['material']
                shape = objects[j]['shape']

                if i == idx_frame - n_his * frame_offset:
                    attrs.append(encode_attr(
                        material, shape, self.bbox_size, self.args.attr_dim))

                bbox_xyxy, xyhw_exp, crop_box, crop_box_v2 = decode_mask_to_box(objects[j]['mask'],\
                        [self.bbox_size, self.bbox_size], self.H, self.W)
                #img_crop = normalize(crop(img, crop_box, self.H, self.W), 0.5, 0.5).permute(2, 0, 1)
                img_crop = normalize(crop(img, crop_box_v2, self.H, self.W), 0.5, 0.5).permute(2, 0, 1)
                tube_id = utilsTube.get_tube_id_from_bbox(bbox_xyxy, frame['frame_index'], self.metadata[idx_video]['tubes'])
                if tube_id==-1:
                    pdb.set_trace()
                if self.args.box_only_flag:
                    xyhw_norm = (xyhw_exp - 0.5)/0.5
                    s = torch.cat([xyhw_norm], 0).unsqueeze(0), tube_id
                else:
                    s = torch.cat([xyhw_exp, img_crop], 0).unsqueeze(0), tube_id
                object_inputs.append(s)

            objs.append(object_inputs)

        attr = torch.cat(attrs, 0).view(
            n_objects, self.args.attr_dim, self.bbox_size, self.bbox_size)

        feats = []
        for x in range(n_objects):
            feats.append(objs[0][x][0])

        for i in range(1, len(objs)):
            for x in range(n_objects):
                for y in range(n_objects):
                    id_x = objs[0][x][1]
                    id_y = objs[i][y][1]
                    if id_x == id_y:
                        feats[x] = torch.cat([feats[x], objs[i][y][0]], 1)

        try:
            feats = torch.cat(feats, 0)
        except:
            print(idx_video, idx_frame)

        #pdb.set_trace()
        ### prepare relation attributes
        n_relations = n_objects * n_objects
        Ra = torch.FloatTensor(
            np.ones((
                n_relations,
                self.args.relation_dim * (self.args.n_his + 2),
                self.bbox_size,
                self.bbox_size)) * -0.5)

        # change to relative position
        relation_dim = self.args.relation_dim
        state_dim = self.args.state_dim
        if self.args.box_only_flag:
            for i in range(n_objects):
                for j in range(n_objects):
                    idx = i * n_objects + j
                    Ra[idx, 1::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                    Ra[idx, 2::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
        else:
            for i in range(n_objects):
                for j in range(n_objects):
                    idx = i * n_objects + j
                    Ra[idx, 1::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                    Ra[idx, 2::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                    Ra[idx, 3::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                    Ra[idx, 4::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w
        label_rel = torch.FloatTensor(np.ones((n_objects * n_objects, 1)) * -0.5)

        '''
        ### change absolute pos to relative pos
        feats[:, state_dim+1::state_dim] = \
                feats[:, state_dim+1::state_dim] - feats[:, 1:-state_dim:state_dim]   # x
        feats[:, state_dim+2::state_dim] = \
                feats[:, state_dim+2::state_dim] - feats[:, 2:-state_dim:state_dim]   # y
        feats[:, 1] = 0
        feats[:, 2] = 0
        '''
        #pdb.set_trace()
        x = feats[:, :-state_dim]
        label_obj = feats[:, -state_dim:]
        label_obj[:, 0] -= feats[:, -2*state_dim+0]
        label_obj[:, 1] -= feats[:, -2*state_dim+1]
        label_obj[:, 2] -= feats[:, -2*state_dim+2]
        label_obj[:, 3] -= feats[:, -2*state_dim+3]
        rel = prepare_relations(n_objects)
        rel.append(Ra[:, :-relation_dim])

        '''
        print(rel[-1][0, 0])
        print(rel[-1][0, 1])
        print(rel[-1][0, 2])
        print(rel[-1][2, 3])
        print(rel[-1][2, 4])
        print(rel[-1][2, 5])
        '''

        # print("attr shape", attr.size())
        # print("x shape", x.size())
        # print("label_obj shape", label_obj.size())
        # print("label_rel shape", label_rel.size())

        '''
        for i in range(n_objects):
            print(objs[0][i][1])
            print(label_obj[i, 1])

        time.sleep(10)
        '''

        return attr, x, rel, label_obj, label_rel

