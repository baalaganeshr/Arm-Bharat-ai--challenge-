"""
Quick Start Guide for Face Mask Detection Training

This script helps you set up and train the face mask detection model.
"""

import os
import sys

def main():
    print("=" * 70)
    print("  FACE MASK DETECTION - QUICK START GUIDE")
    print("=" * 70)
    
    # Check dataset
    dataset_path = "dataset"
    with_mask = os.path.join(dataset_path, "with_mask")
    without_mask = os.path.join(dataset_path, "without_mask")
    
    if os.path.exists(with_mask) and os.path.exists(without_mask):
        with_mask_count = len([f for f in os.listdir(with_mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
        without_mask_count = len([f for f in os.listdir(without_mask) if f.endswith(('.jpg', '.png', '.jpeg'))])
        total = with_mask_count + without_mask_count
        
        if total > 0:
            print(f"\n✓ Dataset found: {total} images")
            print(f"  - With Mask: {with_mask_count} images")
            print(f"  - Without Mask: {without_mask_count} images")
            print("\nYou're ready to train!")
        else:
            print("\n⚠ Dataset folders exist but are empty")
            show_dataset_instructions()
    else:
        print("\n⚠ Dataset not found")
        show_dataset_instructions()
    
    # Show next steps
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    
    if total > 0:
        print("\n1. Start Docker Container:")
        print("   docker compose up -d")
        print("\n2. Enter the container:")
        print("   docker exec -it mask-detection-dev bash")
        print("\n3. Train the model:")
        print("   python /app/train_improved.py")
        print("\n4. Or run detection (requires pre-trained model):")
        print("   python /app/detect_final.py")
    else:
        print("\n1. Download dataset first (see instructions above)")
        print("2. Then run this script again")
    
    print("\n" + "=" * 70)

def show_dataset_instructions():
    print("\n" + "-" * 70)
    print("DOWNLOAD DATASET:")
    print("-" * 70)
    print("\nDataset: Face Mask Detection Dataset")
    print("Source: https://www.kaggle.com/omkargurav/face-mask-dataset")
    print("\nOption 1: Manual Download")
    print("  1. Visit the Kaggle link above")
    print("  2. Click 'Download' (requires Kaggle account)")
    print("  3. Extract the zip file")
    print("  4. Copy 'with_mask' and 'without_mask' folders to:")
    print("     ./dataset/")
    print("\nOption 2: Using Kaggle API")
    print("  1. Set up Kaggle API credentials (~/.kaggle/kaggle.json)")
    print("  2. Run: kaggle datasets download -d omkargurav/face-mask-dataset")
    print("  3. Extract and copy folders to ./dataset/")
    print("\nAlternative: Use a sample dataset")
    print("  You can use any face mask detection dataset with the structure:")
    print("  dataset/")
    print("  ├── with_mask/    (images of people wearing masks)")
    print("  └── without_mask/ (images of people not wearing masks)")

if __name__ == "__main__":
    main()
