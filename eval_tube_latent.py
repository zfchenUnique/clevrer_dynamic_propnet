import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import gzip
import pickle
import h5py
import json

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from models import PropagationNetwork

from utils import count_parameters
from utils import prepare_relations, convert_mask_to_bbox, crop, encode_attr
from utils import normalize, check_attr, get_identifier, get_identifiers
from utils import check_same_identifier, check_same_identifiers, check_contain_id
from utils import get_masks, check_valid_masks, check_duplicate_identifier
from utils import rand_float, init_stat, combine_stat, load_data, store_data
from utils import decode, make_video, Tee
import copy
import pdb
import utils_tube as utilsTube 
from data_tube import decode_mask_to_box 
from PIL import Image

utilsTube.set_debugger()

parser = argparse.ArgumentParser()
parser.add_argument('--pn', type=int, default=1)
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--env', default='CLEVR')
parser.add_argument('--dt', type=float, default=1./5.)
parser.add_argument('--train_valid_ratio', type=float, default=0.90909)
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--nf_relation', type=int, default=128)
parser.add_argument('--nf_particle', type=int, default=128)
parser.add_argument('--nf_effect', type=int, default=128*4)
parser.add_argument('--filter_cnter', type=int, default=10)

parser.add_argument('--st_idx', type=int, default=0)
parser.add_argument('--ed_idx', type=int, default=0)
parser.add_argument('--video', type=int, default=0)
parser.add_argument('--store_img', type=int, default=0)
parser.add_argument('--back_ground', default='background.png')

parser.add_argument('--H', type=int, default=100)
parser.add_argument('--W', type=int, default=150)
parser.add_argument('--bbox_size', type=int, default=24)
parser.add_argument('--data_dir', default='frames')
parser.add_argument('--label_dir', default='derender_proposals')
parser.add_argument('--des_dir', default='propnet_predictions')
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--eval_type', default='valid', help='rollout|valid')
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--edge_superv', type=int, default=1, help='whether to include edge supervision')
parser.add_argument('--use_attr', type=int, default=1, help='whether using attributes or not')

parser.add_argument('--eval_part', default='valid')

parser.add_argument('--n_his', type=int, default=2)
parser.add_argument('--frame_offset', type=int, default=5)

parser.add_argument('--what_if', type=int, default=-1)

# object attributes:
# [rubber, metal, cube, cylinder, sphere]
parser.add_argument('--attr_dim', type=int, default=5)

# object state:
# [x, y, w, h, r, g, b]
parser.add_argument('--state_dim', type=int, default=7)

# relation:
# [collision, dx, dy, dw, dh]
parser.add_argument('--relation_dim', type=int, default=5)

# new parameters
parser.add_argument('--tube_dir', default='')
parser.add_argument('--prp_dir', default='')
parser.add_argument('--ann_dir', default='')
parser.add_argument('--eval_full_path', default='')
parser.add_argument('--tube_mode', type=int, default=0)

# dynamic nscl model 
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--desc', required=True, type='checked_file', metavar='FILE')
parser.add_argument('--log_path', type=str, default='dumps/logs')
parser.add_argument('--nscl_path', type=str, default='/home/zfchen/code/nsclClevrer/dynamicNSCL')
parser.add_argument('--rel_box_flag', type=int, default=0)
parser.add_argument('--dynamic_ftr_flag', type=int, default=0)
parser.add_argument('--scene_add_supervision', type=int, default=0)
parser.add_argument('--colli_ftr_type', type=int, default=1, help='0 for average rgb, 1 for KNN sampling')
parser.add_argument('--smp_coll_frm_num', type=int, default=32)
parser.add_argument('--version', type=str, default='v0')
parser.add_argument('--box_iou_for_collision_flag', type=int, default=1)
parser.add_argument('--dataset', required=True, choices=['clevrer'], help='dataset')
parser.add_argument('--box_only_for_collision_flag', type=int, default=0)
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model (default: none)')
parser.add_argument('--img_size', type=int, default=256)

args = parser.parse_args()

def run_main():
    use_gpu = torch.cuda.is_available()
    if args.env == 'CLEVRER':
        #args.n_rollout = 11000
        args.n_rollout = 200
        args.time_step = 100
    else:
        raise AssertionError("Unsupported env")

    args.outf = args.outf + '_' + args.env
    if args.use_attr == 0:
        args.outf += '_noAttr'
        args.des_dir += '_noAttr'
    if args.edge_superv == 0:
        args.outf += '_noEdgeSuperv'
        args.des_dir += '_noEdgeSuperv'
    if args.pn:
        args.outf += '_pn'
    #args.outf += '_pstep_' + str(args.pstep)
    args.outf += '_pstep_' + str(args.pstep)+ '_tubemode_' + str(args.tube_mode)
    # args.dataf = args.dataf + '_' + args.env
    args.evalf = args.evalf + '_' + args.env
    if args.use_attr == 0:
        args.evalf += '_noAttr'
    if args.edge_superv == 0:
        args.evalf += '_noEdgeSuperv'

    os.system('mkdir -p ' + args.evalf)
    os.system('mkdir -p ' + args.des_dir)

    model_nscl = utilsTube.build_nscl_model(args, logger)

    # define interaction network
    model = PropagationNetwork(args, residual=True, use_gpu=use_gpu)
    print("model #params: %d" % count_parameters(model))

    if args.eval_full_path !='':
        model_path = args.eval_full_path
    elif args.epoch == 0 and args.iter == 0:
        model_path = os.path.join(args.outf, 'net_best.pth')
    else:
        model_path = os.path.join(args.outf, 'tube_net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))

    print("Loading saved ckp from %s" % model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterionMSE = nn.MSELoss()
    criterionSL1 = nn.SmoothL1Loss()

    model.cuda()
    model_nscl.cuda()

    test_list = np.arange(args.st_idx, args.ed_idx).tolist()
    n_his = args.n_his
    frame_offset = args.frame_offset
    bbox_size = args.bbox_size
    H = args.H
    W = args.W

    for test_idx in range(len(test_list)):

        print("[%d/%d]" % (test_idx, len(test_list)))
        
        test_idx2 = test_list[test_idx]
        des_pred = dict()
        des_path = os.path.join(args.des_dir, 'sim_%05d.json' % test_list[test_idx])
                
        vid = int(test_idx2/1000)
        ann_full_dir = os.path.join(args.ann_dir, 'annotation_%02d000-%02d000'%(vid, vid+1))
        #pk_path = os.path.join(args.tube_dir, 'annotation_%05d.pk' % test_idx2)
        pk_path = os.path.join(args.tube_dir, 'proposal_%05d.pk' % test_idx2)
        prp_path = os.path.join(args.prp_dir, 'proposal_%05d.json' % test_idx2)
        ann_path = os.path.join(ann_full_dir, 'annotation_%05d.json' % test_idx2)
        if not os.path.isfile(pk_path):
            pk_path = os.path.join(args.tube_dir, 'annotation_%05d.pk' % test_idx2)

        tubes_info = utilsTube.pickleload(pk_path)
        prp_info = utilsTube.jsonload(prp_path)
        data = utilsTube.jsonload(ann_path)
        data['tubes'] = tubes_info['tubes']
        data['proposals'] = prp_info 

        ids_cnter = []

        for i in range(len(data['tubes'])):
            tube_box_list = data['tubes'][i]
            valid_box_num = 0
            for box in tube_box_list:
                if box!=[0, 0, 1, 1]:
                    valid_box_num +=1
            ids_cnter.append([i, valid_box_num])

        ids_filter = []
        for i in range(len(ids_cnter)):
            if ids_cnter[i][1] >= args.filter_cnter:
                ids_filter.append(ids_cnter[i][0])

        frames_gt = []
        for i in range(0, len(data['proposals']['frames']), frame_offset):
            objects = data['proposals']['frames'][i]['objects']
            n_objects = len(objects)

            frame_filename = os.path.join('video_'+str(test_idx2).zfill(5), str(i+1)+'.png') 
            vid = int(test_idx2/1000)
            ann_full_dir = os.path.join(args.data_dir, 'image_%02d000-%02d000'%(vid, vid+1))

            img = cv2.imread(os.path.join(ann_full_dir, frame_filename))
            img = cv2.resize(img, (args.W, args.H), interpolation=cv2.INTER_AREA) / 255.

            frame_objs = []
            frame_rels = []

            ids_cur_frame = []
            
            obj_id_to_map_id  = utilsTube.mapping_obj_ids_to_tube_ids(objects, data['tubes'], i)
            #pdb.set_trace()
            for j in range(n_objects):
                
                bbox_xyxy, xyhw_exp, crop_box, crop_box_v2 = decode_mask_to_box(objects[j]['mask'],\
                            [args.bbox_size, args.bbox_size], H, W)
                tube_id = obj_id_to_map_id[j]
                if tube_id==-1:
                    pass 
                    #pdb.set_trace()
                if not (tube_id in ids_filter):
                    continue 
                
                material = objects[j]['material']
                shape = objects[j]['shape']
                attr = encode_attr(material, shape, bbox_size, args.attr_dim)
                img_crop = normalize(crop(img, crop_box_v2, H, W), 0.5, 0.5).permute(2, 0, 1)
                s = [attr, torch.cat([xyhw_exp, img_crop], 0).unsqueeze(0), tube_id]
                frame_objs.append(s)

            frames_gt.append([frame_objs, frame_rels, None])

        if args.video:
            path = os.path.join(args.evalf, '%d_gt' % test_list[test_idx])
            utilsTube.make_video_from_tube_ann_latent(path, frames_gt, H, W, bbox_size, args.back_ground, args.store_img)
        
        ##### prediction from the learned model
        des_pred['predictions'] = []

        if args.use_attr == 1:
            what_if_ed_idx = len(ids_filter)
        else:
            #what_if_ed_idx = 0
            what_if_ed_idx = len(ids_filter)

        for what_if in range(-1, what_if_ed_idx):
            frames_pred = []

            #if what_if>0:
            #    pdb.set_trace()

            what_if_shown_up = False
            for i in range(0, len(data['proposals']['frames']), frame_offset):

                idx = i // frame_offset

                frame_objs = []
                frame_rels = []
                frame_feats = []

                objs_gt = frames_gt[idx][0]
                rels_gt = frames_gt[idx][1]

                for j in range(len(objs_gt)):
                    tube_id = objs_gt[j][2]

                    valid_idx = False
                    for k in range(len(ids_filter)):
                        if tube_id == ids_filter[k]:

                            if k == what_if:
                                what_if_shown_up = True
                                continue

                            valid_idx = True
                            break

                    if not valid_idx:
                        continue

                    already_considered = True
                    if idx < n_his + 1:
                        already_considered = False
                    else:
                        for k in range(n_his + 1):
                            objs_considering = frames_pred[idx - k - 1][0]

                            contain_id = False
                            for t in range(len(objs_considering)):
                                if tube_id == objs_considering[t][2]:
                                    contain_id = True
                                    break
                            if not contain_id:
                                already_considered = False
                                break

                    if not already_considered:
                        frame_objs.append(objs_gt[j])

                # st_time = time.time()
                if what_if == -1 or not what_if_shown_up:
                    objs_gt = frames_gt[idx][0]
                else:
                    objs_gt = None

                objs_pred, rels_pred, feats_pred = forward_step(frames_pred[idx-n_his-1:idx], model, objs_gt)
                # print(time.time() - st_time)
                frame_objs += objs_pred
                frame_rels += rels_pred
                frame_feats += feats_pred

                frames_pred.append([frame_objs, frame_rels, frame_feats])

            # rollout for extra 12 frames
            # if what_if == -1:
            #pdb.set_trace()
            st_idx = len(frames_pred)
            for idx in range(st_idx, st_idx + 12):
                objs_pred, rels_pred, feats_pred = forward_step(frames_pred[idx-n_his-1:idx], model)
                frames_pred.append([objs_pred, rels_pred, feats_pred])

            traj_predict = dict()

            if what_if == -1:
                video_name = '%d_pred' % test_list[test_idx]
                traj_predict['what_if'] = -1
            else:
                video_name = '%d_pred_epoch%d_iter%d_%d' % (test_list[test_idx], args.epoch, args.iter, what_if)
                traj_predict['what_if'] = what_if

            traj_predict['trajectory'] = []
            traj_predict['collisions'] = []

            for i in range(len(frames_pred)):
                objs, rels, feats = frames_pred[i]
                frame = dict()
                frame['frame_index'] = i * args.frame_offset
                
                if args.use_attr:
                    frame['objects'] = []
                    frame['objects'].append(obj_pred)
                else:
                    frame['feats'] = []
                    for j in range(len(feats)):
                        frame['feats'].append(feats[j].numpy().tolist())

                    frame['objects'] = []
                    frame['imgs'] = []
                    frame['ids'] = []

                    for j in range(len(objs)):
                        obj = copy.deepcopy(objs[j][1][0])
                        id = objs[j][2]
                        x = obj[0, 0, 0] 
                        y = obj[1, 0, 0] 
                        w = obj[2, 0, 0] 
                        h = obj[3, 0, 0] 
                        x -= w/2.0
                        y -= h/2.0


                        obj_pred = dict()
                        obj_pred['x'] = float(x)
                        obj_pred['y'] = float(y)
                        obj_pred['h'] = float(h)
                        obj_pred['w'] = float(w)

                        img = obj[4:].permute(1, 2, 0).data.numpy()
                        img = np.clip((img * 0.5 + 0.5)*255, 0, 255)
                        img = img.astype(np.uint8).tolist()
                        frame['imgs'].append(img)
                        frame['objects'].append(obj_pred)
                        frame['ids'].append(id)
                        #if j==0:
                        #    print('%d_%d_%d\n' %(what_if, i, j))
                        #    print(obj_pred)

                traj_predict['trajectory'].append(frame)

                if args.use_attr:
                    for j in range(len(rels)):
                        id_0, id_1 = rels[j][0], rels[j][1]
                        collide = dict()
                        collide['frame'] = i * args.frame_offset
                        collide['objects'] = []

                        obj_collide = dict()
                        obj_collide['color'] = id_0[0]
                        obj_collide['material'] = id_0[1]
                        obj_collide['shape'] = id_0[2]
                        collide['objects'].append(obj_collide)

                        obj_collide = dict()
                        obj_collide['color'] = id_1[0]
                        obj_collide['material'] = id_1[1]
                        obj_collide['shape'] = id_1[2]
                        collide['objects'].append(obj_collide)

                        traj_predict['collisions'].append(collide)

            des_pred['predictions'].append(traj_predict)

            if args.video:
                path = os.path.join(args.evalf, video_name)
                utilsTube.make_video_from_tube_ann(path, frames_pred, H, W, bbox_size, args.back_ground, args.store_img)
               # pdb.set_trace()
        #pdb.set_trace()

        with open(des_path, 'w') as f:
            json.dump(des_pred, f)

if __name__=='__main__':
    run_main()

