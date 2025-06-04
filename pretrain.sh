#!/bin/bash

TrainType='body'     # 'expression'   'pose'
level=3
transl_weight=5

PosePath="/home/bingxing2/ailab/group/ai4earth/haochen/dataset/pose_dataset/anchor_trainposes"
NAME="${TrainType}_NewLR_WeightedLoss_trans_extraloss${level}_translweight${transl_weight}_multilevel_len100mask40_onlyglobal_0602"

LOG_DIR="/home/bingxing2/ailab/group/ai4earth/haochen/models/logs/${NAME}/"
mkdir -p -- "$LOG_DIR"

CKPT="${LOG_DIR}/checkpoint-500.pth"

module unload compilers/gcc
module unload cudnn
module load cuda/11.7.0 compilers/gcc/9.3.0 cudnn/8.6.0.163_cuda11.x
module unload compilers/gcc
module load compilers/gcc/11.3.0

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /home/bingxing2/ailab/scxlab0094/.conda/envs/chenhao_env/bin/python -m torch.distributed.launch --nproc_per_node=4  new_pretrain.py \
    --log_dir ${LOG_DIR} \
    --output_dir ${LOG_DIR} \
    --batch_size 1 \
    --mode 'pretrain' \
    --data_path ${PosePath} \
    --fixed_number 100 \
    --train_part ${TrainType} \
    --norm_pix_loss \
    --mask_ratio 40 \
    --start_index 30 \
    --extra_edge \
    --level ${level} \
    --grad_weight 0.5 \
    --transl_weight ${transl_weight} \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --emb_dim 1024 --dec_emb_dim 512 --num_heads 8 --decoder_num_heads 8 --depth 10 --decoder_depth 10 \
    > ${LOG_DIR}train_1.log 2>&1 &


    # --extra_edge \

    # --resume ${CKPT} \