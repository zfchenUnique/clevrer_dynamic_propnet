GPU_ID=$1
CLEVRER_ANN_PATH=../../data/raw_data/clevrer
TUBE_PRP_PATH=../../data/raw_data/train_val_proposals 
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_tube.py \
--gen_valid_idx 1 \
--edge_superv 0 \
--use_attr 0 \
--num_workers 1 \
--tube_mode 1 \
--batch_size 1 \
--box_only_flag 1 \
--frame_offset 4 \
--relation_dim 5 \
--state_dim 4 \
--outf dumps/offset5_matchNoIoU \
--data_dir ${CLEVRER_ANN_PATH} \
--prp_dir ${CLEVRER_ANN_PATH}/proposals \
--ann_dir ${CLEVRER_ANN_PATH} \
--tube_dir ${TUBE_PRP_PATH} \
