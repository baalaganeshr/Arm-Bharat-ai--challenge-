"""
Complete Training and Dashboard Runner
Runs training and then starts the web dashboard
"""

import os
import sys
import subprocess

print("=" * 70)
print("  FACE MASK DETECTION - COMPLETE SETUP & TRAINING")
print("=" * 70)

# Step 1: Verify dataset
print("\n[1/4] Verifying dataset...")
with_mask_path = "dataset/with_mask"
without_mask_path = "dataset/without_mask"

with_mask_count = len([f for f in os.listdir(with_mask_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
without_mask_count = len([f for f in os.listdir(without_mask_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
total = with_mask_count + without_mask_count

print(f"  With mask: {with_mask_count} images")
print(f"  Without mask: {without_mask_count} images")
print(f"  Total: {total} images")

if total == 0:
    print("✗ ERROR: No dataset found!")
    sys.exit(1)

print("✓ Dataset verified")

# Step 2: Create directories
print("\n[2/4] Creating output directories...")
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
print("✓ Directories ready")

# Step 3: Run training
print("\n[3/4] Starting training with MobileNetV2...")
print("=" * 70)

result = subprocess.run([sys.executable, "train_advanced.py"], 
                       capture_output=False)

if result.returncode == 0:
    print("\n✓ Training completed successfully!")
else:
    print("\n✗ Training failed!")
    sys.exit(1)

# Step 4: Create sample compliance log for dashboard
print("\n[4/4] Setting up dashboard...")
import pandas as pd
from datetime import datetime, timedelta

# Create sample compliance data
log_file = "logs/compliance_log.csv"
if not os.path.exists(log_file):
    print("Creating sample compliance log...")
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    data = {
        'Timestamp': [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        'With_Mask': [50 + i * 2 for i in range(24)],
        'Without_Mask': [20 - i for i in range(24)],
        'Compliance_Percent': [70 + i for i in range(24)]
    }
    df = pd.DataFrame(data)
    df.to_csv(log_file, index=False)
    print(f"✓ Sample log created: {log_file}")

print("\n" + "=" * 70)
print("  STARTING WEB DASHBOARD")
print("=" * 70)
print("\n  Dashboard URL: http://localhost:9000")
print("  Press Ctrl+C to stop\n")
print("=" * 70)

# Start dashboard
os.chdir("our_improvements")
subprocess.run([sys.executable, "dashboard_app.py"])
