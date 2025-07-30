#!/bin/bash

# Full ISIC 2024 Training Pipeline
# Trains EVA -> EdgeNext -> Ensemble in sequence

set -e  # Exit on any error

echo "=========================================="
echo "ISIC 2024 Full Training Pipeline"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Change to source directory
cd src

# Step 1: Train EVA model
echo "Step 1/3: Training EVA model..."
echo "Started at: $(date)"
mamba run -n isic-2024 python train_eva_edgenext.py --model EVA
echo "EVA training completed at: $(date)"
echo ""

# Step 2: Train EdgeNext model  
echo "Step 2/3: Training EdgeNext model..."
echo "Started at: $(date)"
mamba run -n isic-2024 python train_eva_edgenext.py --model EDGENEXT
echo "EdgeNext training completed at: $(date)"
echo ""

# Step 3: Train ensemble
echo "Step 3/3: Training ensemble model..."
echo "Started at: $(date)"
mamba run -n isic-2024 python train_ensemble.py
echo "Ensemble training completed at: $(date)"
echo ""

echo "=========================================="
echo "FULL PIPELINE COMPLETED!"
echo "End time: $(date)"
echo "=========================================="

# Show final results
if [ -f "submission.csv" ]; then
    echo ""
    echo "Submission file created successfully!"
    echo "Running quick analysis..."
    mamba run -n isic-2024 python submission_analysis.py --quiet
else
    echo "Warning: submission.csv not found!"
fi