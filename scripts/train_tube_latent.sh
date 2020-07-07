GPU_ID=$1
jac-crun ${GPU_ID} python train_tube_latent.py \
--gen_valid_idx 1 \
--edge_superv 0 \
--use_attr 0 \
--data_dir /home/zfchen/code/nsclClevrer/clevrer/ \
--prp_dir /home/zfchen/code/nsclClevrer/clevrer/proposals \
--ann_dir /home/zfchen/code/nsclClevrer/clevrer \
--tube_mode 1 \
--batch_size 1 \
--desc /home/zfchen/code/nsclClevrer/dynamicNSCL/clevrer/desc_nscl_derender_clevrer_v2.py \
--rel_box_flag 0 --dynamic_ftr_flag 1 --version v3 \
--dataset clevrer \
--num_workers 2 \
--data_ver v3 \
--ckp_per_iter 100000 \
--colli_ftr_only 1 \
--norm_ftr_flag 1 \
--lr 0.0001 \
--frame_offset 5 \
--n_his 2 \
--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/frm_31_epoch_24.pth \
--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
--pred_normal_num 22 \
--lam_collision 1.0 \
--visualize_flag 1 \
--residual_rela_pred 0 \
--residual_obj_pred 0 \
--outf dumps/prpOri_latent_norm_ftr_n_his_2 \
--visualize_flag 1 \
--debug 1 \
--obj_spatial_only 1 \
--rela_spatial_only 1 \
--state_dim 4 \
--relation_dim 260 \
--visual_folder dumps/visualization/tube_gt_obj_spatial_only_rela_spatial_only_objPred_res \
--resume_model_full_path dumps/prpOri_latent_norm_ftr_n_his_2_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_version_v3/net_best.pth
#--resume_epoch 1 \
#--resume_iter 400000 \
#--outf dumps/gtResPred_latent_norm_ftr_n_his_2 \
#--outf dumps/gtResPred_latent_norm_ftr_n_his_2 \
#--outf dumps/gt_latent_norm_ftr_n_his_2_box_only_lam1 \
#--outf dumps/gt_latent_norm_ftr_n_his_2_box_only_lam1 \
#--tube_dir ../clevrer/tubeProposalsGt \
#--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/refine_epoch_10.pth \
#--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposals/1.0_1.0 \
#--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposals/1.0_1.0 \
#--debug 1
#--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
#--resume_model_full_path latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_tubemode_1/tube_net_epoch_0_iter_500000.pth \
#--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/frm_31_epoch_24.pth \
#--debug 1
#CUDA_VISIBLE_DEVICES=${GPU_ID} python train_tube_latent.py \
