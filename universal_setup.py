"""
Universal Setup Script
Installs all dependencies and runs the project with or without Docker
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        return True
    except:
        return False

def check_docker():
    """Check if Docker is available"""
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        return True
    except:
        return False

print("=" * 70)
print("  FACE MASK DETECTION - UNIVERSAL SETUP")
print("=" * 70)

# Step 1: Install Python dependencies
print("\n[1/3] Installing Python dependencies...")
dependencies = [
    "tensorflow",
    "opencv-python",
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "flask",
    "pillow"
]

failed = []
for package in dependencies:
    print(f"  Installing {package}...", end=" ")
    if install_package(package):
        print("✓")
    else:
        print("✗")
        failed.append(package)

if failed:
    print(f"\n⚠ Failed to install: {', '.join(failed)}")
    print("Please install manually: pip install " + " ".join(failed))
else:
    print("\n✓ All dependencies installed!")

# Step 2: Check environment
print("\n[2/3] Checking environment...")
HAS_DOCKER = check_docker()

if HAS_DOCKER:
    print("✓ Docker is available")
    USE_DOCKER = True
else:
    print("⚠ Docker not found - will run in local Python")
    USE_DOCKER = False

# Check dataset
dataset_path = "dataset"
with_mask = os.path.join(dataset_path, "with_mask")
without_mask = os.path.join(dataset_path, "without_mask")

if os.path.exists(with_mask) and os.path.exists(without_mask):
    with_count = len([f for f in os.listdir(with_mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    without_count = len([f for f in os.listdir(without_mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if with_count > 0 or without_count > 0:
        print(f"✓ Dataset found: {with_count + without_count} images")
    else:
        print("⚠ Dataset folders empty - please add images")
else:
    print("⚠ Dataset not found - creating folders...")
    os.makedirs(with_mask, exist_ok=True)
    os.makedirs(without_mask, exist_ok=True)

# Step 3: Create necessary directories
print("\n[3/3] Setting up directories...")
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
print("✓ Directories ready")

print("\n" + "=" * 70)
print("  SETUP COMPLETE!")
print("=" * 70)

# Display next steps
print("\n📌 NEXT STEPS:")
print("=" * 70)

if USE_DOCKER:
    print("\n1. Start Docker Container:")
    print("   docker compose up -d")
    print("\n2. Train Model:")
    print("   docker exec -it mask-detection-dev python /app/train_advanced.py")
    print("\n3. Start Dashboard:")
    print("   docker exec -d mask-detection-dev python /app/our_improvements/dashboard_app.py")
    print("   Open: http://localhost:9000")
    print("\n4. Run Detection:")
    print("   docker exec -it mask-detection-dev python /app/realtime_detection_dashboard.py")
else:
    print("\n1. Train Model:")
    print("   python train_advanced.py")
    print("\n2. Start Dashboard (in separate terminal):")
    print("   cd our_improvements")
    print("   python dashboard_app.py")
    print("   Open: http://localhost:9000")
    print("\n3. Run Detection (in separate terminal):")
    print("   python realtime_detection_dashboard.py")

print("\n" + "=" * 70)
