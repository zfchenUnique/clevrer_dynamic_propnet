CUDA_VISIBLE_DEVICES=1 python eval_abs.py \
    --store_img 1 \
    --video 1 \
	--video 1 \
    --des_dir dumps/propnet_ \
    --label_dir /home/zfchen/code/nsclClevrer/clevrer/annotationNew \
    --st_idx 00 \
    --ed_idx 15000 \
    --data_dir /home/zfchen/code/nsclClevrer/clevrer/image_00000-01000 \
    --debug 1 \
    --use_attr 0 \
    --edge_superv 0 \
    --box_only_flag 0 \
    --rm_mask_state_flag 0 \
    --add_hw_state_flag 0 \
    --add_xyhw_state_flag 0 \
    --evalf dumps/dumps/visualization/original \
    --outf dumps/original \
    #--evalf dumps/dumps/visualization/box_only \
    #--outf dumps/box_only \
    #--state_dim 4 \
    #--epoch 2 \
    #--iter 800000 \
    #--outf dumps/original \
    #--data_dir /home/zfchen/code/nsclClevrer/clevrer/image_10000-11000 \
    #--epoch 2 \
    #--iter 400000 \
		

