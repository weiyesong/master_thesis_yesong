#!/bin/bash
# Quick start script for DOFA + TorchGeo + Lightning-UQ-Box training

echo "=========================================="
echo "DOFA EuroSAT Training - Quick Start"
echo "=========================================="
echo ""

# Check if pretrained weights exist
if [ ! -f "checkpoints/DOFA_ViT_base_e100.pth" ]; then
    echo "⚠️  Pretrained weights not found!"
    echo "   Please run: python test_dofa_classifier.py first"
    exit 1
fi

echo "✓ Pretrained weights found"
echo ""

# Run training with default settings
echo "Starting training with default settings:"
echo "  - Epochs: 5"
echo "  - Batch size: 32"
echo "  - Learning rate: 1e-4"
echo "  - Model: DOFA-base"
echo ""

python train_uqbox.py \
    --data_root ./data \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_epochs 5 \
    --num_workers 4 \
    --gpus 1 \
    --experiment_name "dofa_base_default"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "View results with:"
echo "  tensorboard --logdir ./results"
