GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_tube.py \
--gen_valid_idx 1 \
--edge_superv 0 \
--use_attr 0 \
--data_dir /home/zfchen/code/nsclClevrer/clevrer/ \
--prp_dir /home/zfchen/code/nsclClevrer/clevrer/proposals \
--ann_dir /home/zfchen/code/nsclClevrer/clevrer \
--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
--num_workers 4 \
--tube_mode 1 \
--batch_size 1 \
--outf dumps/box_only_tubeProposalAttrV3 \
--debug 0 \
--box_only_flag 1 \
--state_dim 4 \
--relation_dim 3 \
#--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposals/1.0_1.0 \
#--resume_epoch 3 \
#--resume_iter 700000 \
