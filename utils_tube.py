import pickle
import json
import sys
import pycocotools.mask as mask
import copy
import pycocotools.mask as cocoMask
import numpy as np
import torch
import os
from utils import merge_img_patch  
import cv2
import pdb

def sort_by_x(obj):
    return obj[1][0, 1, 0, 0]

def make_video_from_tube_ann(filename, frames, H, W, bbox_size, back_ground=None, store_img=False):

    n_frame = len(frames)

    # print('states', states.shape)
    # print('actions', actions.shape)
    # print(filename)

    # print(actions[:, 0, :])
    # print(states[:20, 0, :])

    videoname = filename + '.avi'
    #videoname = filename + '.mp4'
    os.system('mkdir -p ' + filename)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16}

    colors = [np.array([255,160,122]),
              np.array([224,255,255]),
              np.array([216,191,216]),
              np.array([255,255,224]),
              np.array([245,245,245]),
              np.array([144,238,144])]

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(videoname, fourcc, 3, (W, H))

    if back_ground is not None:
        bg = cv2.imread(back_ground)
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)

    pdb.set_trace()

    for i in range(n_frame):
        objs, rels, feats = frames[i]
        n_objs = len(objs)

        if back_ground is not None:
            frame = bg.copy()
        else:
            frame = np.ones((H, W, 3), dtype=np.uint8) * 255

        objs = objs.copy()


        #pdb.set_trace()
        # obj: attr, [x, y, h, w, img_crop], id
        objs.sort(key=sort_by_x)
        n_object = len(objs)
        for j in range(n_object):
            obj = objs[j][1][0]

            img = obj[4:].permute(1, 2, 0).data.numpy()
            img = np.clip((img * 0.5 + 0.5)*255, 0, 255)
            # img *= mask

            n_rels = len(rels)
            collide = False
            for k in range(n_rels):
                id_0, id_1 = rels[k][0], rels[k][1]
                if id_0==objs[j][2] or id_1==objs[j][2]:
                    collide = True

            if collide:
                _, cont, _ = cv2.findContours(
                    mask.astype(np.uint8)[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cont, -1, (0, 255, 0), 1)

                '''
                print(i, j)
                cv2.imshow('mask', mask.astype(np.uint8))
                cv2.imshow('img', img.astype(np.uint8))
                cv2.waitKey(0)
                '''

            if np.isnan(obj[1, 0, 0]) or np.isnan(obj[2, 0, 0]):
                # check if the position is NaN
                continue
            if np.isinf(obj[1, 0, 0]) or np.isinf(obj[2, 0, 0]):
                # check if the position is inf
                continue

            # differences between crop box and xyxy!!!
            #y = int(obj[0, 0, 0] * W - bbox_size/2)
            #x = int(obj[1, 0, 0] * H - bbox_size/2)
            y = int(obj[0, 0, 0] * W - obj[2, 0, 0] * W /2)
            x = int(obj[1, 0, 0] * H - obj[3, 0, 0] * H /2)

            # print(x, y, H, W)
            h, w = int(bbox_size), int(bbox_size)
            x_ = max(-x, 0)
            y_ = max(-y, 0)
            x = max(x, 0)
            y = max(y, 0)
            h_ = min(h - x_, H - x)
            w_ = min(w - y_, W - y)

            # print(x, y, x_, y_, h_, w_)

            if x + h_ < 0 or x >= H or y + w_ < 0 or y >= W:
                continue

            frame[x:x+h_, y:y+w_] = merge_img_patch(
                frame[x:x+h_, y:y+w_], img[x_:x_+h_, y_:y_+w_])

        if store_img:
            cv2.imwrite(os.path.join(filename, 'img_%d.png' % i), frame.astype(np.uint8))
        # cv2.imshow('img', frame.astype(np.uint8))
        # cv2.waitKey(0)

        out.write(frame)



def decode_mask_to_box(mask, crop_box_size, H, W):
    bbx_xywh = cocoMask.toBbox(mask)
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


    ret = np.ones((4, crop_box_size[0], crop_box_size[1]))
    ret[0, :, :] *= bbx_xywh[0]
    ret[1, :, :] *= bbx_xywh[1]
    ret[2, :, :] *= bbx_xywh[2]
    ret[3, :, :] *= bbx_xywh[3]
    ret = torch.FloatTensor(ret)
    return bbx_xyxy, ret, crop_box.astype(int)   


def mapping_obj_ids_to_tube_ids(objects, tubes, frm_id ):
    obj_id_to_map_id = {}
    fix_ids = []
    for obj_id, obj_info in enumerate(objects):
        bbox_xyxy, xyhw_exp, crop_box = decode_mask_to_box(objects[obj_id]['mask'], [24, 24], 100, 150)
        tube_id = get_tube_id_from_bbox(bbox_xyxy, frm_id, tubes)
        obj_id_to_map_id[obj_id] = tube_id
        if tube_id==-1:
            fix_ids.append(obj_id)

    if len(fix_ids)>0:
        fix_id = 0 # fixiong bugs invalid ids
        for t_id in range(len(tubes)):
            if t_id in obj_id_to_map_id.values():
                continue
            else:
                obj_id_to_map_id[fix_ids[fix_id]] = t_id  
                fix_id  +=1
                print('invalid tube ids!\n')
                if fix_id==len(fix_ids):
                    break 
    tube_id = len(tubes)
    for obj_id, tube_id in obj_id_to_map_id.items():
        if tube_id==-1:
            obj_id_to_map_id[obj_id] = tube_id 
            tube_id +=1
    return obj_id_to_map_id 

def check_box_in_tubes(objects, idx, tubes):

    tube_frm_boxes = [tube[idx] for tube in tubes]
    for obj_id, obj_info in enumerate(objects):
        box_xyxy = decode_box(obj_info['mask'])
        if list(box_xyxy) not in tube_frm_boxes:
            return False
    return True

def decode_box(obj_info):
    bbx_xywh = mask.toBbox(obj_info)
    bbx_xyxy = copy.deepcopy(bbx_xywh)
    bbx_xyxy[2] =  bbx_xyxy[2] + bbx_xyxy[0]
    bbx_xyxy[3] =  bbx_xyxy[3] + bbx_xyxy[1]
    return bbx_xyxy 

def set_debugger():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

def get_tube_id_from_bbox(bbox_xyxy, frame_id, tubes):
    for tube_id, tube_info in enumerate(tubes):
        if tube_info[frame_id]==list(bbox_xyxy):
            return tube_id
    return -1

def get_tube_id_from_bbox(bbox_xyxy, frame_id, tubes):
    for tube_id, tube_info in enumerate(tubes):
        if tube_info[frame_id]==list(bbox_xyxy):
            return tube_id
    return -1

def checking_duplicate_box_among_tubes(frm_list, tubes):
    """
    checking boxes that are using by different tubes
    """
    valid_flag=False
    for frm_idx, frm_id in enumerate(frm_list):
        for tube_id, tube_info in enumerate(tubes):
            tmp_box = tube_info[frm_id] 
            for tube_id2 in range(tube_id+1, len(tubes)):
                if tmp_box==tubes[tube_id2][frm_id]:
                    valid_flag=True
                    return valid_flag
    return valid_flag 

def check_object_inconsistent_identifier(frm_list, tubes):
    """
    checking whether boxes are lost during the track
    """
    valid_flag = False
    for tube_id, tube_info in enumerate(tubes):
        if tube_info[frm_list[0]]!=[0,0,1,1]:
            for tmp_id in range(1, len(frm_list)):
                tmp_frm = frm_list[tmp_id]
                if tube_info[tmp_frm]==[0, 0, 1, 1]:
                    valid_flag=True
                    return valid_flag 
    return valid_flag 

def jsonload(path):
    f = open(path)
    this_ans = json.load(f)
    f.close()
    return this_ans

def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f)
    f.close()

def pickleload(path):
    f = open(path, 'rb')
    this_ans = pickle.load(f)
    f.close()
    return this_ans

def pickledump(path, this_dic):
    f = open(path, 'wb')
    this_ans = pickle.dump(this_dic, f)
    f.close()

