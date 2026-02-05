#!/bin/bash

# ManyDepth Ablation Study Script
# This script runs comprehensive ablation studies to evaluate different components
# of the ManyDepth model architecture and training procedure.

# Set common parameters
DATA_PATH="/mnt/e/cityscapes/"
LOG_BASE_DIR="outs/ablation_studies2"

# Create base log directory
mkdir -p "$LOG_BASE_DIR"

echo "Starting ManyDepth Ablation Studies..."
echo "Data path: $DATA_PATH"
echo "Log directory: $LOG_BASE_DIR"
echo "=========================================="

# No register tokens
echo "5.1 Training without register tokens..."
CUDA_VISIBLE_DEVICES=1 python manydepth/train.py\
    --data_path "$DATA_PATH" \
    --log_dir "$LOG_BASE_DIR/no_register_tokens" \
    --png \
    --g2s \
    --num_register_tokens 0 \

# 1. BASELINE EXPERIMENTS
echo "1. Running Baseline Experiments..."

# 1.1 Original ManyDepth (full model)
echo "1.1 Training original ManyDepth (full model)..."
CUDA_VISIBLE_DEVICES=1 python manydepth/train.py\
    --data_path "$DATA_PATH" \
    --log_dir "$LOG_BASE_DIR/full_model" \
    --png \
    --g2s \

# 1.2 ManyDepth without LoRA
echo "1.2 Training ManyDepth without LoRA..."
CUDA_VISIBLE_DEVICES=1 python manydepth/train.py\
    --data_path "$DATA_PATH" \
    --log_dir "$LOG_BASE_DIR/baseline_no_lora" \
    --png \
    --g2s \
    --no_lora \

# 2. TEMPORAL FUSION ABLATION
echo "2. Running Temporal Fusion Ablation..."

# 2.1 No temporal fusion
echo "2.1 Training without temporal fusion..."
CUDA_VISIBLE_DEVICES=1 python manydepth/train.py\
    --data_path "$DATA_PATH" \
    --log_dir "$LOG_BASE_DIR/no_temporal_fusion" \
    --png \
    --g2s \
    --no_temporal_fusion \

# # 3. CONSISTENCY LOSS ABLATION
echo "3. Running Consistency Loss Ablation..."

# # 3.1 No consistency loss
echo "3.1 Training without consistency loss..."
CUDA_VISIBLE_DEVICES=1 python manydepth/train.py\
    --data_path "$DATA_PATH" \
    --log_dir "$LOG_BASE_DIR/no_consistency_loss" \
    --png \
    --g2s \
    --no_consistency_loss \

# # 4. DYNAMIC LOSS WEIGHT ABLATION
echo "4. Running Dynamic Loss Weight Ablation..."

# # 4.1 No dynamic loss weight
echo "4.1 Training without dynamic loss weight..."
CUDA_VISIBLE_DEVICES=1  python manydepth/train.py\
    --data_path "$DATA_PATH" \
    --log_dir "$LOG_BASE_DIR/no_loss_dynamic_weight" \
    --png \
    --g2s \
    --no_loss_dynamic_weight \


echo "=========================================="
echo "Training completed!"
echo "Results saved in: $LOG_BASE_DIR"
echo "=========================================="
