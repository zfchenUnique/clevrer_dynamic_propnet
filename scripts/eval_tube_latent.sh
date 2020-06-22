GPU_ID=$1
CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_tube_latent.py \
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
    --epoch 0 \
    --video 0 \
    --iter 500000 \
    --tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
    --dataset clevrer \
    --tube_mode 1 \
    --outf latent_prp \
    --desc /home/zfchen/code/nsclClevrer/dynamicNSCL/clevrer/desc_nscl_derender_clevrer_v2.py \
    --rel_box_flag 0 --dynamic_ftr_flag 1 --version v2 \
    --dataset clevrer \
    --frame_offset 5 \
    --n_his 2 \
    --tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
    --load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/frm_31_epoch_24.pth \
		

