GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_abs.py \
--gen_valid_idx 1 \
--edge_superv 1 \
--data_dir /home/zfchen/code/nsclClevrer/clevrer \
--label_dir /home/zfchen/code/nsclClevrer/clevrer/annotationNew \
--num_workers 1 \
--debug 1 \
--box_only_flag 1 \
--state_dim 4 \
--use_attr 0 \
--lam_collision 0 \
--lam_position 1.0 \
--log_per_iter 100 \
#--resume_epoch 2 \
#--resume_iter 300000
