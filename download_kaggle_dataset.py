"""
Download and prepare Face Mask Dataset using Kaggle API
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

def check_kaggle_setup():
    """Check if Kaggle API is properly configured"""
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_path.exists():
        print("✓ Kaggle API credentials found")
        return True
    else:
        print("✗ Kaggle API credentials not found")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Move downloaded kaggle.json to:")
        print(f"   {kaggle_path.parent}")
        return False

def download_dataset():
    """Download dataset from Kaggle"""
    print("\n" + "=" * 70)
    print("DOWNLOADING FACE MASK DATASET FROM KAGGLE")
    print("=" * 70)
    
    if not check_kaggle_setup():
        return False
    
    dataset_name = "omkargurav/face-mask-dataset"
    download_path = "dataset_download"
    
    print(f"\nDownloading {dataset_name}...")
    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset_name,
            "-p", download_path,
            "--unzip"
        ]
        subprocess.run(cmd, check=True)
        print("✓ Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        return False
    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Install with: pip install kaggle")
        return False

def prepare_dataset():
    """Move dataset to correct location"""
    print("\nPreparing dataset...")
    
    # Create dataset directories
    os.makedirs("dataset/with_mask", exist_ok=True)
    os.makedirs("dataset/without_mask", exist_ok=True)
    
    # Check for downloaded files
    download_path = Path("dataset_download")
    if not download_path.exists():
        print("✗ Download folder not found")
        return False
    
    # Find and move image folders
    # The structure might vary, so we'll look for with_mask and without_mask folders
    for root, dirs, files in os.walk(download_path):
        if "with_mask" in root.lower():
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    src = Path(root) / file
                    dst = Path("dataset/with_mask") / file
                    src.rename(dst)
        elif "without_mask" in root.lower():
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    src = Path(root) / file
                    dst = Path("dataset/without_mask") / file
                    src.rename(dst)
    
    # Count images
    with_mask_count = len(list(Path("dataset/with_mask").glob("*.jpg"))) + \
                      len(list(Path("dataset/with_mask").glob("*.png")))
    without_mask_count = len(list(Path("dataset/without_mask").glob("*.jpg"))) + \
                         len(list(Path("dataset/without_mask").glob("*.png")))
    
    print(f"\n✓ Dataset prepared!")
    print(f"  - With Mask: {with_mask_count} images")
    print(f"  - Without Mask: {without_mask_count} images")
    print(f"  - Total: {with_mask_count + without_mask_count} images")
    
    return True

def main():
    print("=" * 70)
    print("  FACE MASK DATASET DOWNLOADER")
    print("=" * 70)
    
    if download_dataset():
        prepare_dataset()
        print("\n✓ All done! You can now train the model.")
        print("\nNext steps:")
        print("1. docker compose up -d")
        print("2. docker exec -it mask-detection-dev bash")
        print("3. python /app/train_improved.py")
    else:
        print("\n✗ Dataset download failed.")
        print("\nManual download instructions:")
        print("1. Visit: https://www.kaggle.com/omkargurav/face-mask-dataset")
        print("2. Download and extract")
        print("3. Copy folders to ./dataset/")

if __name__ == "__main__":
    main()
