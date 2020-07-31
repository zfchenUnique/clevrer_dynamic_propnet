GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_tube.py \
--gen_valid_idx 1 \
--edge_superv 0 \
--use_attr 0 \
--data_dir /home/zfchen/code/nsclClevrer/clevrer/ \
--prp_dir /home/zfchen/code/nsclClevrer/clevrer/proposals \
--ann_dir /home/zfchen/code/nsclClevrer/clevrer \
--num_workers 4 \
--tube_mode 1 \
--batch_size 1 \
--outf dumps/tubeNetAttrV3True_offset4 \
--box_only_flag 0 \
--frame_offset 4 \
--relation_dim 5 \
--state_dim 7 \
--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsAttrV3/1.0_1.0_0.4_0.7 \
--resume_epoch 0 \
--resume_iter 400000 \
#--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposals/1.0_1.0 \
#--debug 1 \
#--debug 1 \
#--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
#--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposals/1.0_1.0 \
