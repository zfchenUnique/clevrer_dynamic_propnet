CUDA_VISIBLE_DEVICES=2 python train.py \
--gen_valid_idx 1 \
--edge_superv 0 \
--data_dir /home/zfchen/code/nsclClevrer/clevrer/image_00000-01000 \
--label_dir /home/zfchen/code/nsclClevrer/clevrer/new_annotation \
--num_workers 0 \
--resume_epoch 0 \
--resume_iter 0
