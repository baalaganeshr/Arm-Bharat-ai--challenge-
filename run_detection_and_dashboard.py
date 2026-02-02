"""
Master Script: Run Detection and Dashboard Together
Starts real-time detection and web dashboard simultaneously
"""

import subprocess
import sys
import os
import time
import threading

print("=" * 70)
print("  FACE MASK DETECTION - INTEGRATED SYSTEM")
print("=" * 70)

# Check if model exists
model_paths = [
    'models/mask_detector_mobilenet.h5',
    'models/mask_detector_128x128_final.h5',
    'models/best_model.h5'
]

model_found = False
for model_path in model_paths:
    if os.path.exists(model_path):
        print(f"✓ Model found: {model_path}")
        model_found = True
        break

if not model_found:
    print("\n✗ No trained model found!")
    print("Please wait for training to complete or run: python train_advanced.py")
    response = input("\nStart training now? (y/n): ")
    if response.lower() == 'y':
        print("\nStarting training...")
        subprocess.run([sys.executable, "train_advanced.py"])
    else:
        sys.exit(1)

# Create sample log if doesn't exist
print("\n[INFO] Preparing dashboard data...")
subprocess.run([sys.executable, "create_sample_log.py"], capture_output=True)

def run_dashboard():
    """Run dashboard in separate thread"""
    print("\n[DASHBOARD] Starting web dashboard...")
    time.sleep(2)  # Give detection time to start first
    os.chdir("our_improvements")
    subprocess.run([sys.executable, "dashboard_app.py"])

def run_detection():
    """Run real-time detection"""
    print("\n[DETECTION] Starting real-time detection...")
    subprocess.run([sys.executable, "realtime_detection_dashboard.py"])

print("\n" + "=" * 70)
print("  SYSTEM STARTING")
print("=" * 70)
print("\n  Real-time Detection: Processing video and logging data")
print("  Web Dashboard: http://localhost:9000")
print("\n  Press Ctrl+C to stop both processes")
print("=" * 70)

# Start dashboard in separate thread
dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()

print("\n✓ Dashboard starting...")
time.sleep(3)  # Give dashboard time to start

print("✓ Starting detection...")

# Run detection in main thread
try:
    run_detection()
except KeyboardInterrupt:
    print("\n\n[INFO] System shutdown by user")
finally:
    print("\n" + "=" * 70)
    print("  SYSTEM STOPPED")
    print("=" * 70)
    print("  Data saved to: logs/compliance_log.csv")
    print("  Dashboard logs: Check terminal output")
    print("=" * 70)
