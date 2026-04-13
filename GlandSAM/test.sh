#!/bin/bash
# Usage:
#   bash test.sh <GPU_ID> <DIR_CHECKPOINT> [CKPT_NAME]
#
# Examples:
#   bash test.sh 0 2D-SAM_vit_b_decoder_lora_glas_seg_noprompt
#   bash test.sh 0 2D-SAM_vit_b_decoder_lora_glas_seg_noprompt checkpoint_best.pth

GPU_ID="${1:?'Error: GPU_ID is required as the first argument'}"
DIR_CHECKPOINT="${2:?'Error: DIR_CHECKPOINT is required as the second argument'}"
CKPT_NAME="${3:-checkpoint_best.pth}"

python test.py \
    -dir_checkpoint "$DIR_CHECKPOINT" \
    -ckpt_name "$CKPT_NAME" \
    -gpu $GPU_ID
