CUDA_VISIBLE_DEVICES=1 python eval_abs.py \
    --store_img 1 \
    --video 1 \
	--video 1 \
    --des_dir dumps/propnet_ \
    --label_dir /home/zfchen/code/nsclClevrer/clevrer/annotationNew \
    --st_idx 00 \
    --ed_idx 15000 \
    --evalf dumps/visulizationNoAttr \
    --data_dir /home/zfchen/code/nsclClevrer/clevrer/image_00000-01000 \
    --outf dumps/box_only \
    --debug 1 \
    --box_only_flag 1 \
    --state_dim 4 \
    --use_attr 0 \
    --edge_superv 0 \
    --epoch 0 \
    --iter 800000 \
    #--data_dir /home/zfchen/code/nsclClevrer/clevrer/image_10000-11000 \
    #--epoch 2 \
    #--iter 400000 \
		

