CUDA_VISIBLE_DEVICES=1 python train.py \
--pstep 1 \
--gen_valid_idx 0 \
--data_dir /home/zfchen/code/nsclClevrer/clevrer/ \
--label_dir /home/zfchen/code/nsclClevrer/clevrer/proposals \
--resume_epoch 1 \
--resume_iter 800000
#--data_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames \
#--label_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/data/derender/processed_proposals \
