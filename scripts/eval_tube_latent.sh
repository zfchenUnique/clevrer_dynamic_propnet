GPU_ID=$1
jac-crun ${GPU_ID} python eval_tube_latent.py \
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
    --evalf dumps/latent_prp \
    --outf dumps/latent_prp_n_his_4_v1 \
    --epoch 1 \
    --video 0 \
    --iter 100000 \
    --tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
    --dataset clevrer \
    --tube_mode 1 \
    --rel_box_flag 0 --dynamic_ftr_flag 1 --version v2 \
    --dataset clevrer \
    --frame_offset 5 \
    --n_his 4 \
    --tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
    --load ../dynamicNSCL/dumps/clevrer/desc_nscl_derender_clevrer_v2/v2_norm_box_even_smp31_col_box_ftr_v2_new_visual_colli_31/checkpoints/epoch_24.pth \
    --desc /userhome/cs/u3004417/code/nsclClevrer/dynamicNSCL/clevrer/desc_nscl_derender_clevrer_v2.py \
    --nscl_path /userhome/cs/u3004417/code/nsclClevrer/dynamicNSCL \
    --data_ver v1 \
    #--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/frm_31_epoch_24.pth \
		

