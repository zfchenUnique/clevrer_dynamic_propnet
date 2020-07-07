CUDA_VISIBLE_DEVICES=1 python eval.py \
    --store_img 1 \
    --video 1 \
    --edge_superv 0 \
    --use_attr 1 \
	--video 1 \
    --des_dir dumps/propnet_ \
    --label_dir /home/zfchen/code/nsclClevrer/clevrer/annotationNew \
    --st_idx 00 \
    --ed_idx 15000 \
    --evalf dumps/visulizationNoAttr \
    --data_dir /home/zfchen/code/nsclClevrer/clevrer/image_00000-01000 \
    #--data_dir /home/zfchen/code/nsclClevrer/clevrer/image_10000-11000 \
    #--epoch 2 \
    #--iter 400000 \
		

