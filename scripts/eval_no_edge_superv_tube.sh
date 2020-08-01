GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_tube.py \
    --store_img 1 \
    --tube_mode 1 \
    --edge_superv 0 \
    --use_attr 0 \
    --des_dir dumps/propnet_predictions_prpAttrV3 \
    --data_dir /home/zfchen/code/nsclClevrer/clevrer \
    --prp_dir /home/zfchen/code/nsclClevrer/clevrer/proposals \
    --ann_dir /home/zfchen/code/nsclClevrer/clevrer \
    --tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsAttrV3/1.0_1.0_0.4_0.7 \
    --box_only_flag 0 \
    --epoch 0 \
    --iter 0 \
    --state_dim 7 \
    --relation_dim 5 \
    --video 0 \
    --des_dir dumps/attrV3_offset4\
    --evalf dumps/visualization_AttrV3_Offset4 \
    --outf dumps/tubeNetAttrV3True_offset4 \
    --epoch 0 \
    --iter 0 \
    --frame_offset 4 \
    --store_patch_flag 1 \
    --st_idx 10000 \
    --ed_idx 15000 \
    #--outf dumps/tubeNetAttrV3True_offset4 \
    #--outf dumps/box_only_tubeGt \
    #--outf dumps/gt \
    #--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
    #--iter 0
    #--epoch 1 \
    #--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsTest/1.0_1.0_0.4_0.7 \
    #--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
    #--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsAttrV3/1.0_1.0_0.4_0.7 \
    #--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposals/1.0_1.0 \
    #--outf file_fixed
    #--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsAttrV0/1.0_1.0\
		

