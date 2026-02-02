"""
Check and Install All Required Libraries
Handles imports gracefully with fallbacks
"""

import sys
import subprocess

def install_and_import(package, import_name=None):
    """Try to import a package, install if missing"""
    if import_name is None:
        import_name = package
    
    try:
        __import__(import_name)
        print(f"✓ {package} already installed")
        return True
    except ImportError:
        print(f"⚠ {package} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            __import__(import_name)
            print(f"✓ {package} installed successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")
            return False

print("=" * 70)
print("  CHECKING AND INSTALLING DEPENDENCIES")
print("=" * 70)
print()

# Core ML/DL
packages = [
    ("tensorflow", "tensorflow"),
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("flask", "flask"),
    ("pillow", "PIL"),
]

all_ok = True
for package, import_name in packages:
    if not install_and_import(package, import_name):
        all_ok = False

print()
print("=" * 70)
if all_ok:
    print("✓ ALL DEPENDENCIES INSTALLED")
else:
    print("⚠ SOME DEPENDENCIES FAILED")
    print("Try running: pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn flask pillow")
print("=" * 70)
