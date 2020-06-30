import os
import sys
import random
import time
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch.multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models_latent import PropagationNetwork
from data_tube_latent import PhysicsCLEVRDataset, collate_fn

from utils import count_parameters, Tee
import pdb
from utils_tube import set_debugger, predict_normal_feature_v2 
from jactorch.cli import escape_desc_name, ensure_path, dump_metainfo
from jacinle.cli.argument import JacArgumentParser
from jacinle.utils.imp import load_source
from jacinle.logging import get_logger, set_output_file
import os.path as osp
import utils_tube 
from jactorch.cuda.copy import async_copy_to
set_debugger()


parser = JacArgumentParser(description='test')
#parser = argparse.ArgumentParser()
parser.add_argument('--pn', type=int, default=1)
parser.add_argument('--pstep', type=int, default=2)

parser.add_argument('--data_dir', default='')
parser.add_argument('--label_dir', default='')

parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--n_particle', type=int, default=0)
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--nf_relation', type=int, default=128)
parser.add_argument('--nf_particle', type=int, default=128)
parser.add_argument('--nf_effect', type=int, default=128*4)
parser.add_argument('--env', default='CLEVRER')
parser.add_argument('--dt', default=1./5.)
parser.add_argument('--train_valid_ratio', type=float, default=0.90909)
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--log_per_iter', type=int, default=500)
parser.add_argument('--ckp_per_iter', type=int, default=50000)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--edge_superv', type=int, default=1, help='whether to include edge supervision')
parser.add_argument('--use_attr', type=int, default=1, help='whether using attributes or not')

parser.add_argument('--n_his', type=int, default=2)
parser.add_argument('--frame_offset', type=int, default=5)
parser.add_argument('--gen_valid_idx', type=int, default=0)

parser.add_argument('--lam_mask', type=float, default=0.2)
parser.add_argument('--lam_position', type=float, default=1.0)
parser.add_argument('--lam_image', type=float, default=1.0)
parser.add_argument('--lam_collision', type=float, default=10.0)

parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--forward_times', type=int, default=2)

parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--resume_iter', type=int, default=0)

# object attributes (material, shape):
# [rubber, metal, cube, cylinder, sphere]
parser.add_argument('--attr_dim', type=int, default=5)

# object state:
# [dx, dy, dw, dh, ftr_dim]
parser.add_argument('--state_dim', type=int, default=260)

# relation:
# [collision_ftr, dx, dy, dw, dh]
parser.add_argument('--relation_dim', type=int, default=260)

# tube new  info
parser.add_argument('--tube_dir', default='')
parser.add_argument('--prp_dir', default='')
parser.add_argument('--ann_dir', default='')
parser.add_argument('--tube_mode', type=int, default=0)
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
parser.add_argument('--data_ver', type=str, default='v0')
parser.add_argument('--box_iou_for_collision_flag', type=int, default=1)
parser.add_argument('--dataset', required=True, choices=['clevrer'], help='dataset')
parser.add_argument('--box_only_for_collision_flag', type=int, default=0)
parser.add_argument('--load', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model (default: none)')
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--resume_model_full_path', type='checked_file', default=None, metavar='FILE', help='load the weights from a pretrained model (default: none)')
parser.add_argument('--colli_ftr_only', type=int, default=0)
parser.add_argument('--norm_ftr_flag', type=int, default=0)
parser.add_argument('--rela_spatial_only', type=int, default=0)
parser.add_argument('--obj_spatial_only', type=int, default=0)
parser.add_argument('--residual_rela_prop', type=int, default=0, help='1 for residual encoding for relations')
parser.add_argument('--pred_res_flag', type=int, default=0, help='1 for residual encoding for prediction')
parser.add_argument('--visualize_flag', type=int, default=0, help='1 for visualization')
parser.add_argument('--add_rela_dist_mode', type=int, default=0)
parser.add_argument('--rela_spatial_dim', type=int, default=4)
parser.add_argument('--rela_ftr_dim', type=int, default=256)
parser.add_argument('--pred_normal_num', type=int, default=12, help='number of frames to predict for regularization')
parser.add_argument('--frm_img_path', default='../clevrer') 
parser.add_argument('--residual_rela_pred', type=int, default=0)
parser.add_argument('--residual_obj_pred', type=int, default=0)

args = parser.parse_args()
args.run_name = 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))


def run_main(args):
    logger = get_logger(__file__)
    args.log_file = osp.join(args.log_path, args.run_name + '.log')
    logger.critical('Writing logs to file: "{}".'.format(args.log_file))
    ### deal with issue that dataloader hangs
    cv2.setNumThreads(0)

    if args.env == 'CLEVRER':
        #args.n_rollout = 11000
        args.time_step = 128
        #args.n_rollout = 11000
        if args.debug:
            args.n_rollout = 15
            args.train_valid_ratio = 0.667
            shuffle_flag = False
        else:
            args.n_rollout = 15000
            args.train_valid_ratio = 0.667
            shuffle_flag = True
    else:
        raise AssertionError("Unsupported env")

    args.outf = args.outf + '_' + args.env
    if args.use_attr == 0:
        args.outf += '_noAttr'
    if args.edge_superv == 0:
        args.outf += '_noEdgeSuperv'
    if args.pn:
        args.outf += '_pn'

    args.outf += '_pstep_' + str(args.pstep)+ '_version_' + str(args.data_ver) #+'_bp'

    os.system('mkdir -p ' + args.outf)

    # setup recorder
    #tee = Tee(os.path.join(args.outf, 'train.log'), 'w')
    print(args)

    # generate data
    datasets = {phase: PhysicsCLEVRDataset(args, phase)  for phase in ['train', 'valid']}
    use_gpu = torch.cuda.is_available()
    
    model_nscl = utils_tube.build_nscl_model(args, logger)

    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size,
        shuffle=shuffle_flag if x == 'train' else False,
        num_workers=args.num_workers,
        collate_fn=collate_fn)
        for x in ['train', 'valid']}

    # define propagation network
    model = PropagationNetwork(args, residual=True, use_gpu=use_gpu)

    print("model #params: %d" % count_parameters(model))

    if args.resume_epoch > 0 or args.resume_iter > 0:
        model_path = os.path.join(args.outf, 'tube_net_epoch_%d_iter_%d.pth' % (args.resume_epoch, args.resume_iter))
        print("Loading saved ckp from %s" % model_path)
        model.load_state_dict(torch.load(model_path))
    elif args.resume_model_full_path:
        print("Loading saved ckp from %s" % args.resume_model_full_path)
        model.load_state_dict(torch.load(args.resume_model_full_path))
    # criterion
    criterionMSE = nn.MSELoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

    if use_gpu:
        model_nscl.cuda()
        model = model.cuda()
        criterionMSE = criterionMSE.cuda()


    st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
    best_valid_loss = np.inf

    for epoch in range(st_epoch, args.n_epoch):

        phases = ['train', 'valid'] if args.eval == 0 else ['valid']
        for phase in phases:
            #if phase == 'train':
            #    continue 

            model.train( phase=='train' and (not args.visualize_flag))

            losses = 0.
            losses_mask = 0.
            losses_position = 0.
            losses_image = 0.
            losses_collision = 0.
            
            for i, data_tube in enumerate(dataloaders[phase]):
                if use_gpu:
                    data_tube = async_copy_to(data_tube, 0)
                #if args.visualize_flag==1 and 0:
                if args.visualize_flag==1:
                    pred_ftr = predict_normal_feature_v2(model, model_nscl, data_tube, args)
                    continue 

                data = utils_tube.prepare_features_temporal_prediction(model_nscl, data_tube, args) 
                attr, x, rel, label_obj, label_rel = data
                node_r_idx, node_s_idx, Ra = rel[3], rel[4], rel[5]
                Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

                Rr = torch.sparse.FloatTensor(
                    Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)]))
                Rs = torch.sparse.FloatTensor(
                    Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)]))

                data = [attr, x, Rr, Rs, Ra, label_obj, label_rel]


                with torch.set_grad_enabled(phase=='train' and (not args.visualize_flag)):
                    for d in range(len(data)):
                        if data[d] is None:
                            continue 
                        if use_gpu:
                                data[d] = data[d].to(x.device)

                    attr, x, Rr, Rs, Ra, label_obj, label_rel = data
                    pred_obj, pred_rel = model(
                        attr, x, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep)
                #pdb.set_trace()
                position = pred_obj[:, :4]
                image = pred_obj[:, 4:]

                '''
                print('mask\n', mask)
                print('x\n', position[0])
                print('y\n', position[1])
                print('img\n', image[0])
                '''
                #pdb.set_trace()
                loss_position = criterionMSE(position, label_obj[:, :4])
                if args.obj_spatial_only==1:
                    loss_image = torch.zeros(1).cuda()
                elif args.norm_ftr_flag:
                    loss_image = criterionMSE(utils_tube._norm(image, dim=1), label_obj[:, 4:])
                else:
                    loss_image = criterionMSE(image, label_obj[:, 4:])
               
                if args.rela_spatial_only==1:
                    box_dim = 4
                    loss_collision = criterionMSE(pred_rel[:, :box_dim], label_rel[:, :box_dim, 0, 0])
                elif args.colli_ftr_only:
                    box_dim = 4
                    if args.norm_ftr_flag:
                        loss_collision = criterionMSE(utils_tube._norm(pred_rel[:, box_dim:], dim=1), label_rel[:, box_dim:])
                    else:
                        loss_collision = criterionMSE(pred_rel[:, box_dim:], label_rel[:, box_dim:])
                else:
                    loss_collision = criterionMSE(pred_rel, label_rel)
                
                loss = loss_position * args.lam_position
                if args.obj_spatial_only!=1:
                    loss += loss_image * args.lam_image
                loss += loss_collision * args.lam_collision

                losses_position += np.sqrt(loss_position.item())
                losses_image += np.sqrt(loss_image.item())
                losses += np.sqrt(loss.item())
                losses_collision += np.sqrt(loss_collision.item())
                losses += np.sqrt(loss.item())

                if phase == 'train':
                    if i % args.forward_times == 0:
                        if i != 0:
                            loss_acc /= args.forward_times
                            optimizer.zero_grad()
                            loss_acc.backward()
                            optimizer.step()
                        loss_acc = loss
                    else:
                        loss_acc += loss

                if i % args.log_per_iter == 0:
                    log = '%s [%d/%d][%d/%d] Pos: %.6f, Obj: %.6f, Col: %.6f, total: %.6f, Agg: %.6f %.6f %.6f %.6f' % \
                          (phase, epoch, args.n_epoch, i, len(dataloaders[phase]),
                           np.sqrt(loss_position.item()),
                           np.sqrt(loss_image.item()),
                           np.sqrt(loss_collision.item()),
                           np.sqrt(loss.item()),
                           losses_position / (i + 1),
                           losses_image / (i + 1),
                           losses_collision / (i + 1),
                           losses / (i + 1))

                    print(log)

                if phase == 'train' and i > 0 and i % args.ckp_per_iter == 0:
                    torch.save(model.state_dict(), '%s/tube_net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i))

            losses /= len(dataloaders[phase])
            log = '%s [%d/%d] Loss: %.4f, Best valid: %.4f' % \
                  (phase, epoch, args.n_epoch, losses, best_valid_loss)
            print(log)

            if phase == 'valid':
                scheduler.step(losses)
                if(losses < best_valid_loss):
                    best_valid_loss = losses
                    torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))

if __name__=='__main__':
    run_main(args)

