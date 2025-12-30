#!/bin/bash
# Quick test script for main.py

echo "=========================================="
echo "Testing main.py setup"
echo "=========================================="
echo ""

# Run a quick test with 1 epoch
python main.py \
    --data_root ./data \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_epochs 1 \
    --num_workers 0 \
    --gpus 0 \
    --experiment_name "test_run" \
    --output_dir ./test_results

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
