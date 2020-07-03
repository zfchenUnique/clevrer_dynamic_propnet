GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_abs.py \
--gen_valid_idx 1 \
--data_dir /home/zfchen/code/nsclClevrer/clevrer \
--label_dir /home/zfchen/code/nsclClevrer/clevrer/annotationNew \
--num_workers 1 \
--use_attr 0 \
--lam_collision 0 \
--lam_position 1.0 \
--lam_hw 1.0 \
--log_per_iter 500 \
--edge_superv 0 \
--box_only_flag 0 \
--add_hw_state_flag 0 \
--add_xyhw_state_flag 1 \
--state_dim 8 \
--rm_mask_state_flag 0 \
--outf dumps/add_xyhw \
#--debug 1 \
#--resume_epoch 2 \
#--resume_iter 300000
