CUDA_VISIBLE_DEVICES=1 python eval.py \
    --store_img 1 \
    --video 1 \
    --edge_superv 0 \
    --use_attr 1 \
	--video 1 \
    --des_dir propnet_attr \
    --data_dir /home/zfchen/code/nsclClevrer/clevrer/image_00000-01000 \
    --label_dir /home/zfchen/code/nsclClevrer/clevrer/new_annotation \
    --st_idx 0 \
    --ed_idx 150 \
    # --epoch 5 \
    # --iter 800000 \
		

