GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_tube.py \
    --store_img 0 \
    --video 0 \
    --tube_mode 1 \
    --edge_superv 0 \
    --use_attr 0 \
    --des_dir propnet_predictions_v1.0 \
    --data_dir /home/zfchen/code/nsclClevrer/clevrer \
    --prp_dir /home/zfchen/code/nsclClevrer/clevrer/proposals \
    --ann_dir /home/zfchen/code/nsclClevrer/clevrer \
    --st_idx 10000 \
    --ed_idx 10020 \
    --evalf dumps/prpGt \
    --outf file_prp \
    --epoch 5 \
    --video 0 \
    --tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
    --iter 200000
    #--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposals/1.0_1.0 \
    #--outf file_fixed
    #--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsAttrV0/1.0_1.0\
    #--epoch 0 \
    #--iter 100000 \
		

