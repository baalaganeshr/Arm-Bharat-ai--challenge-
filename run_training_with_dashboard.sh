#!/bin/bash
# Complete Setup and Training Script with Dashboard
# This script runs inside the Docker container

echo "======================================================================"
echo "  FACE MASK DETECTION - COMPLETE SETUP & TRAINING"
echo "======================================================================"

# Step 1: Install/Verify dependencies
echo ""
echo "[1/5] Checking dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r /app/requirements.txt
pip install --quiet flask

echo "✓ Dependencies installed"

# Step 2: Verify dataset
echo ""
echo "[2/5] Verifying dataset..."
WITH_MASK_COUNT=$(find /app/dataset/with_mask -type f | wc -l)
WITHOUT_MASK_COUNT=$(find /app/dataset/without_mask -type f | wc -l)
TOTAL=$((WITH_MASK_COUNT + WITHOUT_MASK_COUNT))

echo "  With mask: $WITH_MASK_COUNT images"
echo "  Without mask: $WITHOUT_MASK_COUNT images"
echo "  Total: $TOTAL images"

if [ $TOTAL -eq 0 ]; then
    echo "✗ ERROR: No dataset found!"
    exit 1
fi

echo "✓ Dataset verified"

# Step 3: Create necessary directories
echo ""
echo "[3/5] Creating output directories..."
mkdir -p /app/models
mkdir -p /app/logs
echo "✓ Directories ready"

# Step 4: Run training
echo ""
echo "[4/5] Starting training with MobileNetV2..."
echo "======================================================================"
python /app/train_advanced.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training completed successfully!"
else
    echo ""
    echo "✗ Training failed!"
    exit 1
fi

# Step 5: Start dashboard
echo ""
echo "[5/5] Starting dashboard..."
echo "======================================================================"
echo ""
echo "  Dashboard will be available at: http://localhost:9000"
echo ""
echo "======================================================================"

cd /app/our_improvements
python dashboard_app.py --host=0.0.0.0 --port=9000
