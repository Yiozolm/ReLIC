#!/bin/bash
set -e

echo "================================================="
echo "Starting ResFlow Training Script"
echo "================================================="

TRAIN_DATA_PATH="your/train/dataset"  
EVAL_DATA_PATH="your/eval/dataset"   
SAVE_DIR="your/checkpoint/dir" 
LOG_DIR="your/log/dir"       

mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR


EPOCHS=100
BATCH_SIZE=8
PATCH_SIZE="256 256" 
NUM_WORKERS=16
CLIP_MAX_NORM=1.0 

N=128 
M=192 
LMBDA=0.0067 
BETA=0.05    

MODEL="elic"
UNET_DIM=64
CFM_MATCHER_TYPE="ot"
CFM_SIGMA=0.1
CFM_SAMPLING_STEPS=50

LR_VAE=1e-4 
LR_CFM=2e-5 

USE_LAZY_INIT=False
INIT_THRESHOLD=10


CHECKPOINT=""

echo "Configuration:"
echo "  - Train Data: ${TRAIN_DATA_PATH}"
echo "  - Eval Data:  ${EVAL_DATA_PATH}"
echo "  - Save Dir:   ${SAVE_DIR}"
echo "  - Log Dir:    ${LOG_DIR}"
echo "  - Model:     ${MODEL}"
echo "  - Epochs:     ${EPOCHS}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Lambda:     ${LMBDA}"
echo "  - Beta:       ${BETA}"
echo "  - VAE LR:     ${LR_VAE}"
echo "  - CFM LR:     ${LR_CFM}"
echo "  - Lazy Init:  ${USE_LAZY_INIT} (Threshold: ${INIT_THRESHOLD} epochs)"
echo "  - Checkpoint: ${CHECKPOINT:-"Not specified"}"
echo "-------------------------------------------------"


python train.py \
    --train_data $TRAIN_DATA_PATH \
    --eval_data $EVAL_DATA_PATH \
    --save_path $SAVE_DIR \
    --log_dir $LOG_DIR \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --patch-size $PATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --lambda $LMBDA \
    --beta $BETA \
    --learning-rate $LR_VAE \
    --learning-rate-cfm $LR_CFM \
    --N $N \
    --M $M \
    --unet-dim $UNET_DIM \
    --cfm-matcher $CFM_MATCHER_TYPE \
    --cfm-sigma $CFM_SIGMA \
    --cfm-sampling-steps $CFM_SAMPLING_STEPS \
    --init-threshold $INIT_THRESHOLD \
    --clip_max_norm $CLIP_MAX_NORM \
    --use_norm \

echo "================================================="
echo "Training script finished."
echo "================================================="