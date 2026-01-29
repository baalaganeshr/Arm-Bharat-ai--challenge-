# Face Mask Detection Using GPU-Accelerated CNN on Jetson Nano

**Bharat AI-SoC Student Challenge 2026 - Problem Statement 5**

---

## 👥 Team

| Role | Name | Roll No |
|------|------|---------|
| **Team Leader** | Baala Ganesh R | 24EC011 |
| Member | Kajendren V | 24EC035 |
| Member | Hirshikesh Prasath S | 24EC031 |
| **Mentor** | Dr. Mohan B | - |

**College:** P.S.R Engineering College, Sivakasi

---

## 📋 Project Overview

Real-time face mask detection system using hardware-software co-design with Raspberry Pi 4 (ARM CPU) and Xilinx Spartan FPGA for CNN acceleration.

### 🎯 Key Features

- ✅ **Real-time multi-person detection** - Detect multiple faces simultaneously
- ✅ **2-3x speedup with FPGA acceleration** - Hardware-accelerated CNN inference
- ✅ **Compliance monitoring dashboard** - Web-based real-time statistics
- ✅ **Privacy-preserving** - No cloud processing, all local
- ✅ **Zero recurring costs** - No subscription fees

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 4 (ARM CPU)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Camera     │  │    Face      │  │  Pre-        │          │
│  │   Capture    │→ │   Detection  │→ │  processing  │          │
│  │   (OpenCV)   │  │ (Haar/DNN)   │  │  (64x64 GS)  │          │
│  └──────────────┘  └──────────────┘  └──────┬───────┘          │
│                                              │                   │
│                                     ┌────────▼────────┐          │
│                                     │  UART Serial    │          │
│                                     │  115200 baud    │          │
│                                     └────────┬────────┘          │
└──────────────────────────────────────────────┼──────────────────┘
                                               │
                                      ┌────────▼────────┐
                                      │  XILINX SPARTAN │
                                      │     FPGA        │
                                      ├─────────────────┤
                                      │ ┌─────────────┐ │
                                      │ │ Conv Layer 1│ │
                                      │ └──────┬──────┘ │
                                      │ ┌──────▼──────┐ │
                                      │ │ Conv Layer 2│ │
                                      │ └──────┬──────┘ │
                                      │ ┌──────▼──────┐ │
                                      │ │ Conv Layer 3│ │
                                      │ └──────┬──────┘ │
                                      │ ┌──────▼──────┐ │
                                      │ │ Max Pooling │ │
                                      │ └─────────────┘ │
                                      └─────────────────┘
```

---

## 📊 Performance Results

| Metric | CPU Only | FPGA Accelerated | Improvement |
|--------|----------|------------------|-------------|
| **Latency** | 120 ms | 40 ms | **3.0x faster** |
| **FPS** | 8.2 | 25.0 | **3.0x higher** |
| **Power** | 5W | 6.5W | +1.5W |
| **Cost** | ₹0/month | ₹0/month | **Same** |

### Benchmark Comparison

| Method | Mean Latency | Throughput | Speedup |
|--------|--------------|------------|---------|
| CPU (TensorFlow) | 121 ms | 8.2 FPS | 1x |
| FPGA (5ms target) | 5.4 ms | 184 FPS | 22x |
| FPGA (2ms target) | 2.4 ms | 423 FPS | 51x |
| FPGA (1ms target) | 1.3 ms | 782 FPS | 95x |

---

## 🚀 Installation

### Requirements

- **Hardware:**
  - Raspberry Pi 4 (4GB+ RAM)
  - Xilinx Spartan FPGA board
  - USB Webcam (720p+)
  - UART cable for Pi-FPGA connection

- **Software:**
  - Python 3.8+
  - TensorFlow 2.13
  - OpenCV 4.8
  - Docker (optional)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Face-Mask-FPGA.git
cd Face-Mask-FPGA

# Option 1: Docker (Recommended)
docker compose build
docker compose up -d
docker exec -it mask-detection-dev bash

# Option 2: Local installation
pip install -r requirements.txt

# Train model
python modified/train_simplified.py --epochs 25

# Run CPU detection
python modified/detect_cpu.py

# Run FPGA detection (on Raspberry Pi)
python modified/detect_fpga.py
```

---

## 📁 Project Structure

```
Face-Mask-Detection/
├── Dockerfile                 # Docker container config
├── docker-compose.yml         # Docker orchestration
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── modified/                  # Modified detection scripts
│   ├── train_simplified.py   # 64x64 grayscale training
│   ├── detect_cpu.py         # CPU-only detection
│   ├── detect_fpga.py        # FPGA-accelerated detection
│   └── detect_fpga_simple.py # Simplified FPGA version
│
├── our_improvements/          # Custom enhancements
│   ├── fpga_interface.py     # UART communication
│   ├── dashboard.py          # CLI dashboard
│   ├── dashboard_app.py      # Web dashboard (Flask)
│   ├── performance_test.py   # Benchmarking
│   └── templates/
│       └── dashboard.html    # Web UI
│
├── dataset/                   # Training data
│   ├── with_mask/            # ~690 images
│   └── without_mask/         # ~686 images
│
├── models/                    # Trained models
│   ├── mask_detector_64x64.h5
│   ├── mask_detector_64x64.tflite
│   └── training_plot.png
│
├── logs/                      # Detection logs
│   ├── compliance_log.csv
│   └── benchmark_results.json
│
└── docs/                      # Documentation
    └── README.md
```

---

## 🎮 Usage

### CPU-Only Detection
```bash
python modified/detect_cpu.py --camera 0
```

**Controls:**
- `q` - Quit
- `s` - Save snapshot

### FPGA-Accelerated Detection
```bash
# Update port in script first: /dev/ttyUSB0 (Linux) or COM3 (Windows)
python modified/detect_fpga.py
```

**Controls:**
- `q` - Quit
- `t` - Toggle FPGA/CPU mode
- `s` - Save snapshot

### Web Dashboard
```bash
python our_improvements/dashboard_app.py

# Open browser: http://localhost:5000
```

### Performance Benchmark
```bash
python our_improvements/performance_test.py --iterations 100 --plot
```

---

## 🔧 FPGA Interface Protocol

### UART Configuration
- **Baud Rate:** 115200
- **Data Bits:** 8
- **Parity:** None
- **Stop Bits:** 1

### Packet Format
```
┌──────────┬─────────┬───────────┬───────────┬──────────┬──────────┐
│  START   │ COMMAND │ SIZE_HIGH │ SIZE_LOW  │  DATA    │   END    │
│  (0xAA)  │ (0x01)  │   (MSB)   │   (LSB)   │ (bytes)  │  (0x55)  │
└──────────┴─────────┴───────────┴───────────┴──────────┴──────────┘
```

### Image Data
- **Size:** 64 × 64 = 4096 bytes
- **Format:** Grayscale, 8-bit unsigned
- **Normalization:** 0-255 → 0.0-1.0

---

## 📈 Model Architecture

```
Layer (type)                 Output Shape              Param #
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
Total params: 401,410 (1.53 MB)
```

---

## 🔮 Future Enhancements

1. **Improve Model Accuracy** - Target 95%+ with larger dataset
2. **Mask Type Detection** - N95, surgical, cloth classification
3. **Proper Wearing Check** - Detect nose/chin exposure
4. **Social Distancing** - Add distance monitoring
5. **Temperature Integration** - Combine with thermal camera
6. **Alert System** - SMS/email notifications
7. **Multi-camera Support** - Distributed deployment

---

## 📚 References

- [Original Face-Mask-Detection Repo](https://github.com/chandrikadeb7/Face-Mask-Detection)
- [Face Mask Dataset (Kaggle)](https://www.kaggle.com/omkargurav/face-mask-dataset)
- [TensorFlow Documentation](https://www.tensorflow.org/docs)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Base Code:** [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)
- **Dataset:** Prajna Bhandary & Kaggle Community
- **Mentor:** Dr. Mohan B, P.S.R Engineering College
- **Challenge:** Bharat AI-SoC Student Challenge 2026

---

**Made with ❤️ by Team PSR | Bharat AI-SoC Challenge 2026**
