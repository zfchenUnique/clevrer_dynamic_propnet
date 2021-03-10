GPU_ID=$1
outf_path=val_release_v2
sep_full_path=../../data/models/dynamic_models/box_only_attrV3.pth
eval_full_path=../../data/models/dynamic_models/attrV3_offset4.pth
CLEVRER_ANN_PATH=../../data/raw_data/clevrer
TUBE_PRP_PATH=../../data/raw_data/train_val_proposals 
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_tube_sep.py \
    --store_img 1 \
    --tube_mode 1 \
    --edge_superv 0 \
    --use_attr 0 \
    --store_patch_flag 1 \
    --st_idx 15000 \
    --ed_idx 20000 \
    --box_only_flag 0 \
    --state_dim 7 \
    --relation_dim 5 \
    --new_mode 0 \
    --frame_offset 5 \
    --outf dumps/${outf_path} \
    --des_dir dumps/annos/${outf_path}_separate_realOffset5 \
    --evalf dumps/visualization/${outf_path}_separate_realOffset5 \
    --epoch 0 \
    --iter 0 \
    --state_dim_spatial 4 \
    --relation_dim_spatial 3 \
    --separate_mode 1 \
    --maskout_pixel_inference_flag 1 \
    --des_dir dumps/propnet_predictions_prpAttrV3 \
    --data_dir ${CLEVRER_ANN_PATH} \
    --prp_dir ${CLEVRER_ANN_PATH}/proposals \
    --ann_dir ${CLEVRER_ANN_PATH} \
    --tube_dir ${TUBE_PRP_PATH} \
    --eval_spatial_full_path ${sep_full_path} \
    --eval_full_path ${eval_full_path} \
		

