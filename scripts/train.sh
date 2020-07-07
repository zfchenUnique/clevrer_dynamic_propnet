GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_abs.py \
--gen_valid_idx 1 \
--edge_superv 1 \
--data_dir /home/zfchen/code/nsclClevrer/clevrer \
--label_dir /home/zfchen/code/nsclClevrer/clevrer/annotationNew \
--num_workers 1 \
#--resume_epoch 2 \
#--resume_iter 300000
