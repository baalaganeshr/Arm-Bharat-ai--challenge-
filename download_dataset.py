"""
Script to download Face Mask Detection Dataset from Kaggle
Dataset: https://www.kaggle.com/omkargurav/face-mask-dataset
"""

import os
import sys

print("=" * 70)
print("  FACE MASK DATASET SETUP")
print("=" * 70)

dataset_url = "https://www.kaggle.com/omkargurav/face-mask-dataset"

print(f"\nDataset URL: {dataset_url}")
print("\n[INFO] To download the dataset, you have two options:\n")

print("Option 1: Manual Download")
print("-" * 50)
print("1. Visit:", dataset_url)
print("2. Click 'Download' button")
print("3. Extract the zip file")
print("4. Copy 'with_mask' and 'without_mask' folders to:")
print("   c:\\Users\\Admin\\Desktop\\arm bharat\\Arm-Bharat-ai--challenge-\\dataset\\")
print()

print("Option 2: Using Kaggle API")
print("-" * 50)
print("1. Install kaggle: pip install kaggle")
print("2. Setup Kaggle API credentials (kaggle.json)")
print("3. Run: kaggle datasets download -d omkargurav/face-mask-dataset")
print("4. Extract to the dataset/ folder")
print()

# Check if dataset already exists
dataset_path = "dataset"
with_mask = os.path.join(dataset_path, "with_mask")
without_mask = os.path.join(dataset_path, "without_mask")

if os.path.exists(with_mask) and os.path.exists(without_mask):
    with_mask_count = len([f for f in os.listdir(with_mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    without_mask_count = len([f for f in os.listdir(without_mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print("\n[SUCCESS] Dataset found!")
    print(f"  With Mask: {with_mask_count} images")
    print(f"  Without Mask: {without_mask_count} images")
    print(f"  Total: {with_mask_count + without_mask_count} images")
else:
    print("\n[WARNING] Dataset not found. Please download it using one of the options above.")

print("\n" + "=" * 70)
