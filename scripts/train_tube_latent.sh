GPU_ID=$1
jac-crun ${GPU_ID} python train_tube_latent.py \
--gen_valid_idx 1 \
--edge_superv 0 \
--use_attr 0 \
--data_dir /home/zfchen/code/nsclClevrer/clevrer/ \
--prp_dir /home/zfchen/code/nsclClevrer/clevrer/proposals \
--ann_dir /home/zfchen/code/nsclClevrer/clevrer \
--tube_dir /home/zfchen/code/nsclClevrer/clevrer/tubeProposalsGt \
--tube_mode 1 \
--batch_size 1 \
--outf latent_prp \
--desc /home/zfchen/code/nsclClevrer/dynamicNSCL/clevrer/desc_nscl_derender_clevrer_v2.py \
--rel_box_flag 0 --dynamic_ftr_flag 1 --version v2 \
--dataset clevrer \
--load /home/zfchen/code/nsclClevrer/dynamicNSCL/dumps/remote_models/frm_31_epoch_24.pth \
--num_workers 4 \
--resume_epoch 0 \
--resume_iter 500000 \
--debug 1
#CUDA_VISIBLE_DEVICES=${GPU_ID} python train_tube_latent.py \
