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

def check_valid_object_id_list_v2(x, args):
    valid_object_id_list = []
    x_step  = args.n_his + 1
    box_dim = 4
    for obj_id in range(x.shape[0]):
        tmp_obj_feat = x[obj_id].view(x_step, -1)
        obj_valid = True
        for tmp_step in range(x_step):
            last_obj_box = tmp_obj_feat[tmp_step, :box_dim]
            x_c, y_c, w, h = last_obj_box
            x1 = x_c - w*0.5
            y1 = y_c - h*0.5
            x2 = x_c + w*0.5
            y2 = y_c + h*0.5
            if w <=0 or h<=0:
                obj_valid = False
            elif x2<=0 or y2<=0:
                obj_valid = False
            elif x1>=1 or y1>=1:
                obj_valid = False
        if obj_valid:
            valid_object_id_list.append(obj_id)
    return valid_object_id_list 

def prepare_valid_input(x, Ra, valid_object_id_list, args):
    x_valid_list = [x[obj_id] for obj_id in valid_object_id_list]
    x_valid = torch.stack(x_valid_list, dim=0)
    valid_obj_num = len(valid_object_id_list)

    rel = prepare_relations(valid_obj_num)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(x_valid.device)

    n_objects = x.shape[0]
    ra_valid_list = []
    for i in range(n_objects):
        for j in range(n_objects):
            idx = i * n_objects + j
            if (i in valid_object_id_list) and (j in valid_object_id_list):
                ra_valid_list.append(Ra[idx])
    Ra_valid = torch.stack(ra_valid_list, dim=0)

    rel.append(Ra_valid)
    attr = None 
    node_r_idx, node_s_idx, Ra_valid = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)])).to(x_valid.device)
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)])).to(x_valid.device)
    return attr, x_valid, Rr, Rs, Ra_valid, node_r_idx, node_s_idx 


def visualize_prediction_v2(box_ftr, feed_dict, whatif_id=-1, store_img=False, args=None):

    # print('states', states.shape)
    # print('actions', actions.shape)
    # print(filename)

    # print(actions[:, 0, :])
    # print(states[:20, 0, :])
    base_folder = os.path.basename(args.load).split('.')[0]
    filename = str(feed_dict['meta_ann']['scene_index'])
    videoname = 'dumps/'+ base_folder + '/' + filename + '_' + str(int(whatif_id)) +'.avi'
    #videoname = filename + '.mp4'
    if store_img:
        img_folder = 'dumps/'+base_folder +'/'+filename 
        os.system('mkdir -p ' + img_folder)

    background_fn = '../temporal_reasoning-master/background.png'
    if not os.path.isfile(background_fn):
        background_fn = '../temporal_reasoningv2/background.png'
    bg = cv2.imread(background_fn)
    H, W, C = bg.shape
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
    fps = 2
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(videoname, fourcc, fps, (W, H))
    
    scene_idx = feed_dict['meta_ann']['scene_index']
    sub_idx = int(scene_idx/1000)
    sub_img_folder = 'image_'+str(sub_idx).zfill(2)+'000-'+str(sub_idx+1).zfill(2)+'000'
    img_full_folder = os.path.join(args.frm_img_path, sub_img_folder) 

    n_frame =  box_ftr.shape[1]
    padding_patch_list = []
    for i in range(n_frame):
        if whatif_id==-1:
            if i < n_frame:
                #frm_id = feed_dict['tube_info']['frm_list'][i]
                frm_id = args.frame_offset * i
                img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
                img_ori = cv2.imread(img_full_path)
                img = copy.deepcopy(img_ori)
                for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
                    tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    if i==len(feed_dict['tube_info']['frm_list'])-1:
                        padding_patch = img_ori[int(y*H):int(y*H+h*H),int(x*W):int(W*x+w*W)]
                        hh, ww, c = padding_patch.shape
                        if hh*ww*c==0:
                            padding_patch  = np.zeros((24, 24, 3), dtype=np.float32)
                        padding_patch_list.append(padding_patch)
            else:
                break
                #pred_offset =  i - len(feed_dict['tube_info']['frm_list'])
                #frm_id = feed_dict['tube_info'] ['frm_list'][-1] + (args.frame_offset*pred_offset+1)  
                frm_id = args.frame_offset * i
                img = copy.deepcopy(bg)
                for tube_id in range(box_ftr.shape[0]):
                    tmp_box = box_ftr[tube_id][pred_offset]
                    x = float(tmp_box[0] - tmp_box[2]*0.5)
                    y = float(tmp_box[1] - tmp_box[3]*0.5)
                    w = float(tmp_box[2])
                    h = float(tmp_box[3])
                    y2 = y +h
                    x2 = x +w
                    if w<=0 or h<=0:
                        continue
                    if x>1:
                        continue
                    if y>1:
                        continue
                    if x2 <=0:
                        continue
                    if y2 <=0:
                        continue 
                    if x<0:
                        x=0
                    if y<0:
                        y=0
                    if x2>1:
                        x2=1
                    if y2>1:
                        y2=1
                    patch_resize = cv2.resize(padding_patch_list[tube_id], (max(1, int(x2*W) - int(x*W)), max(1, int(y2*H) - int(y*H))) )
                    img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                    img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                    cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
            if store_img:
                cv2.imwrite(os.path.join( img_folder, '%s_%d.png' % (filename, i)), img.astype(np.uint8))
        else:
            #frm_id = feed_dict['tube_info']['frm_list'][i]
            frm_id = args.frame_offset * i
            img_full_path = os.path.join(img_full_folder, 'video_'+str(scene_idx).zfill(5), str(frm_id+1)+'.png')
            img_rgb = cv2.imread(img_full_path)
            #for tube_id in range(len(feed_dict['tube_info']['box_seq']['tubes'])):
            #img = copy.deepcopy(bg)
            img = copy.deepcopy(img_rgb)
            for tube_id in range(box_ftr.shape[0]):
                tmp_box = feed_dict['tube_info']['box_seq']['tubes'][tube_id][frm_id]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                img_patch = img_rgb[int(y*H):int(y*H + h*H) , int(x*W): int(x*W + w*W)]
                hh, ww, c = img_patch.shape
                if hh*ww*c==0:
                    img_patch  = np.zeros((24, 24, 3), dtype=np.float32)
                img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (36,255,12), 1)
                cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                tmp_box = box_ftr[tube_id][i]
                x = float(tmp_box[0] - tmp_box[2]*0.5)
                y = float(tmp_box[1] - tmp_box[3]*0.5)
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                y2 = y +h
                x2 = x +w
                if w<=0 or h<=0:
                    continue
                if x>1:
                    continue
                if y>1:
                    continue
                if x2 <=0:
                    continue
                if y2 <=0:
                    continue 
                if x<0:
                    x=0
                if y<0:
                    y=0
                if x2>1:
                    x2=1
                if y2>1:
                    y2=1
                #patch_resize = cv2.resize(img_patch, (max(int(x2*W) - int(x*W), 1), max(int(y2*H) - int(y*H), 1)))
                #img[int(y*H):int(y2*H), int(x*W):int(x2*W)] = patch_resize
                img = cv2.rectangle(img, (int(x*W), int(y*H)), (int(x*W + w*W), int(y*H + h*H)), (0,0,0), 1)
                cv2.putText(img, str(tube_id), (int(x*W), int(y*H)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            if store_img:
                cv2.imwrite(os.path.join( img_folder, '%s_%d_%d.png' % (filename, i, int(whatif_id))), img.astype(np.uint8))
        out.write(img)


def check_valid_object_id_list(x, args):
    valid_object_id_list = []
    x_step  = args.n_his + 1
    box_dim = 4
    for obj_id in range(x.shape[0]):
        tmp_obj_feat = x[obj_id].view(x_step, -1)
        last_obj_box = tmp_obj_feat[-1, :box_dim]
        x_c, y_c, w, h = last_obj_box
        x1 = x_c - w*0.5
        y1 = y_c - h*0.5
        x2 = x_c + w*0.5
        y2 = y_c + h*0.5
        obj_valid = True
        if w <=0 or h<=0:
            obj_valid = False
        elif x2<=0 or y2<=0:
            obj_valid = False
        elif x1>=1 or y1>=1:
            obj_valid = False
        if obj_valid:
            valid_object_id_list.append(obj_id)
    return valid_object_id_list 


def prepare_normal_prediction_input(feed_dict, f_sng, args, p_id=0):
    """"
    attr: obj_num, attr_dim, 1, 1 (None)
    x: obj_num, state_dim*(n_his+1)
    rel: return from prepare_relations
    label_obj: obj_num, state_dim, 1 , 1
    label_rel: obj_num * obj_num, rela_dim, 1, 1
    """""
    x_step = args.n_his +1
    st_id = p_id
    ed_id = p_id + x_step
    if ed_id >len(feed_dict['tube_info']['frm_list']):
        return None
    first_frm_id_list = [frm_id for frm_id in feed_dict['tube_info']['frm_list'][st_id:ed_id]]
    obj_num, t_dim, box_dim = f_sng[3].shape
    ftr_dim = f_sng[1].shape[-1]
    spatial_seq = f_sng[3].view(obj_num, t_dim, box_dim)
    tmp_box_list = [spatial_seq[:, frm_id] for frm_id in range(st_id, ed_id)]
    x_box = torch.stack(tmp_box_list, dim=1).contiguous().view(obj_num, args.n_his+1, box_dim)  
    x_ftr = f_sng[0][:, st_id:ed_id] .view(obj_num, x_step, ftr_dim)
    if args.obj_spatial_only==1:
        x = x_box.view(obj_num, x_step*box_dim, 1, 1).contiguous()
    else:
        x = torch.cat([x_box, x_ftr], dim=2).view(obj_num, x_step*(ftr_dim+box_dim), 1, 1).contiguous()


    # obj_num*obj_num, box_dim*total_step, 1, 1
    spatial_rela = extract_spatial_relations_v2(x_box.view(obj_num, x_step, box_dim), args)
    ftr_rela = f_sng[2][:, :, st_id:ed_id].view(obj_num*obj_num, x_step*ftr_dim, 1, 1) 
    rela = torch.cat([spatial_rela, ftr_rela], dim=1)
    rel = prepare_relations(obj_num)
    for idx in range(len(rel)-2):
        rel[idx] = rel[idx].to(ftr_rela.device)
    rel.append(rela)
    attr = None 
    node_r_idx, node_s_idx, Ra = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)])).to(ftr_rela.device)
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)])).to(ftr_rela.device)

    return attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx 


def predict_normal_feature_v2(model, model_nscl, feed_dict, args):
    with torch.no_grad():
        f_sng = extract_features(model_nscl, feed_dict)

    data = prepare_normal_prediction_input(feed_dict, f_sng, args)
    #x: obj_num, state_dim*(n_his+1)
    x_step = args.n_his + 1
    attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data
    pred_obj_list = []
    pred_rel_spatial_list = []
    pred_rel_ftr_list = []
    box_dim = 4
    ftr_dim = f_sng[1].shape[-1]
    rela_spa_dim = args.rela_spatial_dim
    rela_ftr_dim = args.rela_ftr_dim
    Ra_spatial = Ra[:, :rela_spa_dim*x_step]
    Ra_ftr = Ra[:, rela_spa_dim*x_step:]
    valid_object_id_stack = []
   
    pred_rel_spatial_gt_list = []
    for t_step in range(args.n_his+1):
        pred_obj_list.append(x[:,t_step*args.state_dim:(t_step+1)*args.state_dim])
        pred_rel_spatial_list.append(Ra_spatial[:, t_step*rela_spa_dim:(t_step+1)*rela_spa_dim]) 
        pred_rel_ftr_list.append(Ra_ftr[:, t_step*ftr_dim:(t_step+1)*ftr_dim]) 

    n_objects_ori = x.shape[0]
    relation_dim = args.relation_dim
    state_dim = args.state_dim 
    for p_id in range(args.pred_normal_num):
        x = torch.cat(pred_obj_list[p_id:p_id+x_step], dim=1) 
        Ra_spatial = torch.cat(pred_rel_spatial_list[p_id:p_id+x_step], dim=1) 
        Ra_ftr = torch.cat(pred_rel_ftr_list[p_id:p_id+x_step], dim=1) 
        Ra = torch.cat([Ra_spatial, Ra_ftr], dim=1)
  
        if p_id>=1:
            data_debug = prepare_normal_prediction_input(feed_dict, f_sng, args, p_id=p_id)
            attr_d, x_d, Rr_d, Rs_d, Ra_d, node_r_idx_d, node_s_idx_d = data_debug
            #x = x_d
            #Ra = Ra_d 
            for t_step  in range(x_step):
                pass 
                #x[:, t_step*state_dim+4: t_step*state_dim+260] =  \
                #        x_d[:, t_step*state_dim+4: t_step*state_dim+260]
                #x[:, t_step*state_dim: t_step*state_dim+4] =  \
                #        x_d[:, t_step*state_dim: t_step*state_dim+4]
            #pdb.set_trace()
        # remove invalid object, object coordinates that has been out of size
        valid_object_id_list = check_valid_object_id_list_v2(x, args) 
        if len(valid_object_id_list) == 0:
            break
        valid_object_id_stack.append(valid_object_id_list)
        data_valid = prepare_valid_input(x, Ra, valid_object_id_list, args)
        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx = data_valid 
        n_objects = x.shape[0]
        feats = x
        invalid_rela_list = []
        # update relation
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 0:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra[idx, 1:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra[idx, 2:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
                Ra[idx, 3:rela_spa_dim*x_step:rela_spa_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w
                if args.add_rela_dist_mode==1 or args.add_rela_dist_mode==2:
                    Ra_x = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                    Ra_y = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                    Ra_dist = torch.sqrt(Ra_x**2+Ra_y**2+0.0000000001) 
                    Ra[idx, 4:rela_spa_dim*x_step:rela_spa_dim] = Ra_dist  
                    
                    if Ra_dist[-1] > args.rela_dist_thre:
                        invalid_rela_list.append(idx)
                    #print(Ra_dist[-1])
        if args.add_rela_dist_mode==2:
            Rr, Rs = update_valid_rela_input(n_objects, invalid_rela_list, feats, args)
        
        # padding spatial relation feature
        pred_rel_spatial_gt = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=Ra.dtype, \
                device=Ra.device) - 1.0
        #pred_rel_spatial_gt[:, 0] = -1
        pred_rel_spatial_gt_valid = Ra[:, (x_step-1)*rela_spa_dim:x_step*rela_spa_dim].squeeze(3).squeeze(2) 
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_spatial_gt[ori_idx] = pred_rel_spatial_gt_valid[valid_idx]
        pred_rel_spatial_gt_list.append(pred_rel_spatial_gt)

        # normalize data
        pred_obj_valid, pred_rel_valid = model(
            attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
       
        pred_obj = torch.zeros(n_objects_ori, state_dim, 1, 1, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) - 1.0
        for valid_id, ori_id in enumerate(valid_object_id_list):
            pred_obj[ori_id] = pred_obj_valid[valid_id]
            pred_obj[ori_id, box_dim:] = _norm(pred_obj_valid[valid_id, box_dim:], dim=0)
        
        pred_rel_ftr = torch.zeros(n_objects_ori*n_objects_ori, ftr_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        pred_rel_spatial = torch.zeros(n_objects_ori*n_objects_ori, rela_spa_dim, dtype=pred_obj_valid.dtype, \
                device=pred_obj_valid.device) #- 1.0
        #pred_rel_spatial[:, 0] = -1
        #pred_rel_spatial[:, 1] = -1
        
        for valid_id, ori_id in enumerate(valid_object_id_list):
            for valid_id_2, ori_id_2 in enumerate(valid_object_id_list):
                valid_idx = valid_id * n_objects + valid_id_2 
                ori_idx = ori_id * n_objects_ori + ori_id_2
                pred_rel_ftr[ori_idx] = _norm(pred_rel_valid[valid_idx, rela_spa_dim:], dim=0)
                pred_rel_spatial[ori_idx] = pred_rel_valid[valid_idx, :rela_spa_dim]

        pred_obj_list.append(pred_obj)
        pred_rel_ftr_list.append(pred_rel_ftr.view(n_objects_ori*n_objects_ori, ftr_dim, 1, 1)) 
        pred_rel_spatial_list.append(pred_rel_spatial.view(n_objects_ori*n_objects_ori, rela_spa_dim, 1, 1)) # just padding
    #pdb.set_trace()
    #make the output consitent with video scene graph
    pred_frm_num = len(pred_obj_list) 
    box_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, :box_dim].contiguous().view(n_objects_ori, pred_frm_num, box_dim) 
    rel_ftr_exp = torch.stack(pred_rel_ftr_list[-pred_frm_num:], dim=1).view(n_objects_ori, n_objects_ori, pred_frm_num, ftr_dim)
    if args.obj_spatial_only==1:
        obj_ftr=None
    else:
        obj_ftr = torch.stack(pred_obj_list[-pred_frm_num:], dim=1)[:, :, box_dim:].contiguous().view(n_objects_ori, pred_frm_num, ftr_dim) 
    if args.visualize_flag:
        visualize_prediction_v2(box_ftr, feed_dict, whatif_id=100, store_img=True, args=args)
        pdb.set_trace()
    return obj_ftr, None, rel_ftr_exp, box_ftr.view(n_objects_ori, -1), valid_object_id_stack, pred_rel_spatial_list, pred_rel_spatial_gt_list     


def _norm(x, dim=-1):
    return x / (x.norm(2, dim=dim, keepdim=True)+1e-7)

def build_nscl_model(args, logger):
    sys.path.append(args.nscl_path)
    from nscl.datasets import initialize_dataset 
    from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
    from jacinle.utils.imp import load_source
    args.desc_name = escape_desc_name(args.desc)
    desc = load_source(args.desc)
    configs = desc.configs
    #args.configs.apply(configs)
    args.configs = configs
    initialize_dataset(args.dataset, args.version)
    logger.critical('Building the model.')
    model = desc.make_model(args)
    if args.load:
        model.load_state_dict(torch.load(args.load)['model'])
        logger.critical('Loaded weights from pretrained model: "{}".'.format(args.load))
    return model

def prepare_relations(n):
    node_r_idx = np.arange(n)
    node_s_idx = np.arange(n)

    rel = np.zeros((n**2, 2))
    rel[:, 0] = np.repeat(np.arange(n), n)
    rel[:, 1] = np.tile(np.arange(n), n)

    # print(rel)

    n_rel = rel.shape[0]
    Rr_idx = torch.LongTensor([rel[:, 0], np.arange(n_rel)])
    Rs_idx = torch.LongTensor([rel[:, 1], np.arange(n_rel)])
    value = torch.FloatTensor([1] * n_rel)

    rel = [Rr_idx, Rs_idx, value, node_r_idx, node_s_idx]

    return rel

def extract_features(model, feed_dict, mode=3):
    with torch.no_grad():
        f_scene = model.resnet(feed_dict['img_future'])
        f_sng = model.scene_graph(f_scene, feed_dict, mode=mode)
        return f_sng 

def extract_spatial_relations_v2(feats, args=None):
    """
    Extract spatial relations
    """
    ### prepare relation attributes
    n_objects, t_frame, box_dim = feats.shape
    feats = feats.view(n_objects, t_frame*box_dim, 1, 1)
    n_relations = n_objects * n_objects
    if args is None or args.add_rela_dist_mode ==0:
        relation_dim =  box_dim
    elif args.add_rela_dist_mode==1 or args.add_rela_dist_mode==2:
        relation_dim =  box_dim + 1
    else:
        raise NotImplementedError 
    state_dim = box_dim 
    Ra = torch.ones([n_relations, relation_dim *t_frame, 1, 1], device=feats.device) * -0.5

    #change to relative position
    #  relation_dim = self.args.relation_dim
    #  state_dim = self.args.state_dim
    for i in range(n_objects):
        for j in range(n_objects):
            idx = i * n_objects + j
            Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
            Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
            Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
            Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w
            if  args is not None and (args.add_rela_dist_mode==1 or args.add_rela_dist_mode==2):
                Ra_x = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
                Ra_y = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
                Ra_dist = torch.sqrt(Ra_x**2+Ra_y**2) #+0.0000000001) 
                Ra[idx, 4::relation_dim] = Ra_dist  
    return Ra

def extract_spatial_relations(feats):
    """
    Extract spatial relations
    """
    ### prepare relation attributes
    n_objects, t_frame, box_dim = feats.shape
    feats = feats.view(n_objects, t_frame*box_dim, 1, 1)
    n_relations = n_objects * n_objects
    relation_dim =  box_dim
    state_dim = box_dim 
    Ra = torch.ones([n_relations, relation_dim *t_frame, 1, 1], device=feats.device) * -0.5

    #change to relative position
    #  relation_dim = self.args.relation_dim
    #  state_dim = self.args.state_dim
    for i in range(n_objects):
        for j in range(n_objects):
            idx = i * n_objects + j
            Ra[idx, 0::relation_dim] = feats[i, 0::state_dim] - feats[j, 0::state_dim]  # x
            Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # y
            Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # h
            Ra[idx, 3::relation_dim] = feats[i, 3::state_dim] - feats[j, 3::state_dim]  # w
    return Ra


def prepare_features_temporal_prediction(model, feed_dict, args=None):
    """"
    attr: obj_num, attr_dim, 1, 1
    x: obj_num, state_dim*(n_his+1)
    rel: return from prepare_relations
    label_obj: obj_num, state_dim, 1 , 1
    label_rel: obj_num * obj_num, rela_dim, 1, 1
    """""
    with torch.no_grad():
        f_sng = extract_features(model, feed_dict)
        obj_num, total_step, ftr_dim = f_sng[1].shape 
        box_dim = f_sng[3].shape[2]
        x_step = total_step - 1
        attr = None
        x_ftr = f_sng[1][:, :x_step].view(obj_num, x_step, ftr_dim, 1, 1)
        x_box = f_sng[3][:, :x_step].view(obj_num, x_step, 4, 1, 1)
        if args is None or args.obj_spatial_only!=1:
            x = torch.cat([x_box, x_ftr], dim=2).view(obj_num, x_step*(ftr_dim+4), 1, 1).contiguous()
        else: 
            x = x_box.view(obj_num, x_step*4, 1, 1).contiguous()
        label_obj_ftr = f_sng[1][:, x_step].view(obj_num, 1, ftr_dim, 1, 1)
        label_obj_box = f_sng[3][:, x_step].view(obj_num, 1, 4, 1, 1)
        if args is None or args.obj_spatial_only!=1:
            label_obj = torch.cat([label_obj_box,  label_obj_ftr], dim=2).view(obj_num, ftr_dim+4, 1, 1).contiguous()
        else:
            label_obj = label_obj_box.view(obj_num, 4, 1, 1).contiguous()
        # obj_num*obj_num, box_dim*total_step, 1, 1
        spatial_rela = extract_spatial_relations(f_sng[3])
        spatial_input = spatial_rela[:, :-box_dim]
        spatial_label = spatial_rela[:, -box_dim:]
        
        ftr_input = f_sng[2][:, :, :x_step].view(obj_num*obj_num, x_step*ftr_dim, 1, 1) 
        ftr_label = f_sng[2][:, :, x_step].view(obj_num*obj_num, ftr_dim, 1, 1) 
        
        rela_input = torch.cat([spatial_input, ftr_input], dim=1)
        label_rel = torch.cat([spatial_label, ftr_label], dim=1)

        rel = prepare_relations(obj_num)
        rel.append(rela_input)
        #pdb.set_trace()
        return attr, x, rel, label_obj, label_rel 

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

    #pdb.set_trace()

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

        store_img = True

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

