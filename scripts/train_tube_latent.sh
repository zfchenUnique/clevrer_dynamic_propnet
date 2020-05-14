GPU_ID=$1
jac-crun ${GPU_ID} python train_tube_latent.py \
--gen_valid_idx 1 \
--edge_superv 0 \
--use_attr 0 \
--data_dir /home/zfchen/code/nsclClevrer/clevrer/ \
--prp_dir /home/zfchen/code/nsclClevrer/clevrer/proposals \
--ann_dir /home/zfchen/code/nsclClevrer/clevrer \
--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
--tube_mode 1 \
--batch_size 1 \
--outf dumps/latent_norm_ftr_001 \
--desc /home/zfchen/code/nsclClevrer/dynamicNSCL/clevrer/desc_nscl_derender_clevrer_v2.py \
--rel_box_flag 0 --dynamic_ftr_flag 1 --version v3 \
--dataset clevrer \
--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/frm_31_epoch_24.pth \
--num_workers 2 \
--data_ver v3 \
--resume_epoch 0 \
--resume_iter 0 \
--ckp_per_iter 100000 \
--colli_ftr_only 1 \
--norm_ftr_flag 1 \
--lr 0.001
#--debug 1
#--resume_model_full_path latent_prp_CLEVRER_noAttr_noEdgeSuperv_pn_pstep_2_tubemode_1/tube_net_epoch_0_iter_500000.pth \
#--debug 1
#CUDA_VISIBLE_DEVICES=${GPU_ID} python train_tube_latent.py \
