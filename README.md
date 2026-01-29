# Face Mask Detection with FPGA Acceleration

A real-time face mask detection system optimized for FPGA deployment, featuring CPU-FPGA hybrid inference and comprehensive compliance monitoring.

## 🚀 Features

- **64x64 Grayscale CNN** - Optimized for FPGA deployment (vs. standard 224x224 RGB)
- **FPGA Acceleration** - UART-based communication with FPGA accelerator
- **CPU Fallback** - Seamless switching between FPGA and CPU inference
- **Real-time Detection** - Live webcam processing with FPS tracking
- **Compliance Dashboard** - CSV logging, statistics, and visualization
- **Docker Support** - Containerized development environment

## 📁 Project Structure

```
Face-Mask-Detection-FPGA/
├── Dockerfile              # TensorFlow GPU container
├── docker-compose.yml      # Container orchestration
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── modified/               # Modified detection scripts
│   ├── train_simplified.py    # 64x64 grayscale training
│   ├── detect_cpu.py          # CPU-only detection
│   └── detect_fpga.py         # FPGA-accelerated detection
│
├── our_improvements/       # Custom enhancements
│   ├── fpga_interface.py      # UART communication
│   ├── dashboard.py           # Compliance tracking
│   └── performance_test.py    # CPU vs FPGA benchmark
│
├── dataset/               # Training data
│   ├── with_mask/
│   └── without_mask/
│
├── models/                # Trained models
│   └── mask_detector_64x64.h5
│
├── logs/                  # Detection logs
│   └── compliance_log.csv
│
└── docs/                  # Documentation
    └── README.md
```

## 🛠️ Quick Start

### Option 1: Docker (Recommended)

```bash
# Build container
docker-compose build

# Start container
docker-compose up -d

# Enter container
docker exec -it mask-detection-dev bash

# Train model
python modified/train_simplified.py --epochs 25

# Run detection
python modified/detect_cpu.py
```

### Option 2: Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset to dataset/ folder

# Train model
python modified/train_simplified.py

# Run detection
python modified/detect_cpu.py
```

## 📊 Training

```bash
# Basic training (25 epochs)
python modified/train_simplified.py

# Custom configuration
python modified/train_simplified.py --epochs 50 --batch_size 64 --img_size 64 --lr 0.0001
```

**Output:**
- `models/mask_detector_64x64.h5` - Keras model
- `models/mask_detector_64x64.tflite` - TFLite model
- `models/training_plot.png` - Training metrics

## 🎥 Detection

### CPU Detection
```bash
python modified/detect_cpu.py --camera 0
```

### FPGA Detection
```bash
# With real FPGA
python modified/detect_fpga.py --port COM3  # Windows
python modified/detect_fpga.py --port /dev/ttyUSB0  # Linux

# With simulator (for testing)
python modified/detect_fpga.py --simulate
```

**Controls:**
- `q` - Quit
- `s` - Save snapshot
- `f` - Toggle FPGA/CPU mode

## 📈 Performance Benchmark

```bash
# Run benchmark
python our_improvements/performance_test.py --iterations 100 --plot

# Results saved to logs/benchmark_results.json
```

## 📋 Compliance Dashboard

```bash
# Generate reports
python our_improvements/dashboard.py --export --plot

# Output:
# - logs/hourly_report.csv
# - logs/daily_report.csv
# - logs/summary_report.json
# - logs/compliance_trend.png
```

## 🔌 FPGA Interface

The FPGA interface uses UART serial communication:

- **Baud Rate:** 115200
- **Image Format:** 64x64 grayscale, flattened bytes
- **Protocol:** START_FRAME → DATA → CHECKSUM → END_FRAME

```python
from our_improvements.fpga_interface import FPGAInterface

# Connect to FPGA
fpga = FPGAInterface(port='COM3', baudrate=115200)

# Process image
result = fpga.process_image(image_64x64)
class_id, confidence = result  # 0=mask, 1=no_mask
```

## 🏗️ Model Architecture

```
Layer (type)                 Output Shape              Params
================================================================
conv2d (Conv2D)              (None, 64, 64, 32)        320
batch_normalization          (None, 64, 64, 32)        128
max_pooling2d                (None, 32, 32, 32)        0
conv2d_1 (Conv2D)            (None, 32, 32, 64)        18,496
batch_normalization_1        (None, 32, 32, 64)        256
max_pooling2d_1              (None, 16, 16, 64)        0
conv2d_2 (Conv2D)            (None, 16, 16, 64)        36,928
batch_normalization_2        (None, 16, 16, 64)        256
max_pooling2d_2              (None, 8, 8, 64)          0
conv2d_3 (Conv2D)            (None, 8, 8, 128)         73,856
batch_normalization_3        (None, 8, 8, 128)         512
max_pooling2d_3              (None, 4, 4, 128)         0
flatten                      (None, 2048)              0
dense (Dense)                (None, 128)               262,272
dropout (Dropout)            (None, 128)               0
dense_1 (Dense)              (None, 64)                8,256
dropout_1 (Dropout)          (None, 64)                0
dense_2 (Dense)              (None, 2)                 130
================================================================
Total params: 401,410
```

## 📦 Dataset

Download the Face Mask Dataset:
- [Kaggle: Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- [GitHub: Prajna Bhandary](https://github.com/prajnasb/observations)

Extract to:
```
dataset/
├── with_mask/      # ~1900 images
└── without_mask/   # ~1900 images
```

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.13+
- OpenCV 4.8+
- CUDA 11.8+ (for GPU training)
- Docker (optional)

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Original repo: [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)
- Dataset: Prajna Bhandary
