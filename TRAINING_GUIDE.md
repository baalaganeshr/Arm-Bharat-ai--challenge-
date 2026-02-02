# Quick Start Guide - Face Mask Detection Training

## 🚀 Current Status

- ✅ Docker configuration ready
- ✅ Dataset directories created
- ⏳ Docker container building...
- ⚠️ Dataset needs to be downloaded

## 📋 Steps to Complete

### Step 1: Download Dataset

You have **3 options** to get the dataset:

#### Option A: Kaggle API (Recommended)
```powershell
# 1. Set up Kaggle API credentials
# Download kaggle.json from https://www.kaggle.com/account
# Place it in C:\Users\Admin\.kaggle\kaggle.json

# 2. Run the download script
python download_kaggle_dataset.py
```

#### Option B: Manual Download
1. Visit https://www.kaggle.com/omkargurav/face-mask-dataset
2. Click "Download" (requires Kaggle account)
3. Extract the downloaded zip file
4. Copy `with_mask` and `without_mask` folders to:
   ```
   c:\Users\Admin\Desktop\arm bharat\Arm-Bharat-ai--challenge-\dataset\
   ```

#### Option C: Alternative Dataset
You can use any face mask detection dataset with this structure:
```
dataset/
├── with_mask/     (images of people wearing masks)
└── without_mask/  (images of people not wearing masks)
```

### Step 2: Wait for Docker Build

The Docker container is currently building. This may take 5-10 minutes.

You can check the build status:
```powershell
docker compose ps
```

### Step 3: Start the Container

Once the build is complete:
```powershell
# Start the container in detached mode
docker compose up -d

# Verify it's running
docker ps
```

### Step 4: Train the Model

Enter the container and start training:
```powershell
# Enter the container
docker exec -it mask-detection-dev bash

# Inside the container, run training
python /app/train_improved.py
```

The training will:
- Load images from `/app/dataset/`
- Train a CNN model for 30 epochs
- Save the best model to `/app/models/`
- Generate training history and plots

### Step 5: Test the Model

After training, you can test detection:
```bash
# Inside the container
python /app/detect_final.py
```

## 📊 Expected Training Output

```
======================================================================
  FACE MASK DETECTION - IMPROVED TRAINING WITH GPU
======================================================================
[SUCCESS] GPU Available: /physical_device:GPU:0
[INFO] Configuration:
  Image Size: 128x128
  Batch Size: 32
  Epochs: 30
  Learning Rate: 0.0001

[INFO] Loading dataset from /app/dataset...
Found X images belonging to 2 classes.
Found Y images belonging to 2 classes.

Epoch 1/30
...
```

## 🔍 Verify Dataset

Before training, verify your dataset:
```powershell
python quick_start.py
```

This will show:
- Number of images in each class
- Dataset structure validation
- Next steps

## 📁 Project Structure

```
Arm-Bharat-ai--challenge-/
├── dataset/              # Training data (you need to add this)
│   ├── with_mask/       # Images of people with masks
│   └── without_mask/    # Images of people without masks
├── models/              # Trained models will be saved here
├── logs/                # Training logs
├── train_improved.py    # Main training script
├── detect_final.py      # Detection script
├── docker-compose.yml   # Docker configuration
└── Dockerfile          # Docker image definition
```

## 🛠️ Helper Scripts

- `quick_start.py` - Check setup status
- `download_dataset.py` - Dataset info
- `download_kaggle_dataset.py` - Auto-download from Kaggle

## ⚠️ Troubleshooting

### Dataset Issues
```powershell
# Check if dataset folders exist and have images
Get-ChildItem dataset/with_mask
Get-ChildItem dataset/without_mask
```

### Docker Issues
```powershell
# Check Docker status
docker --version
docker compose --version

# View build logs
docker compose logs

# Rebuild if needed
docker compose build --no-cache
```

### GPU Issues
If training is slow, you might be running on CPU only. Check:
- NVIDIA drivers installed
- Docker Desktop configured for GPU access
- NVIDIA Container Toolkit installed

## 📝 Notes

- Training time depends on dataset size and GPU availability
- Without GPU, training will be slower but still work
- The trained model will be saved in the `models/` directory
- You can monitor training with TensorBoard (if configured)

## 🎯 Next Steps After Training

1. Convert model to TensorFlow Lite: `python convert_to_tflite.py`
2. Test detection: `python detect_final.py`
3. Deploy to Raspberry Pi: See [docs/README.md](docs/README.md)
4. Set up FPGA acceleration: See FPGA integration docs

---

**Need help?** Check the main [README.md](README.md) for full documentation.
