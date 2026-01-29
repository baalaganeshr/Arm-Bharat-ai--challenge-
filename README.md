# Face Mask Detection with FPGA Acceleration

**Bharat AI-SoC Student Challenge 2026 - Problem Statement 5**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)

---

## 👥 Team Information

| Role | Name | Roll No |
|------|------|---------|
| **Team Leader** | Baala Ganesh R | 24EC011 |
| Member | Kajendren V | 24EC035 |
| Member | Hirshikesh Prasath S | 24EC031 |

- **College:** P.S.R Engineering College, Sivakasi
- **Mentor:** Dr. Mohan B

---

## 📋 Project Overview

Real-time face mask detection using **hardware-software co-design** with Raspberry Pi 4 and Xilinx Spartan FPGA.

### 🎯 Key Features

- ✅ Real-time multi-person detection
- ✅ 3x speedup with FPGA acceleration
- ✅ Web dashboard for monitoring
- ✅ Privacy-preserving (local processing)
- ✅ Zero recurring costs

## 📊 Performance

| Metric | CPU | FPGA | Improvement |
|--------|-----|------|-------------|
| **Latency** | 120ms | 40ms | **3.0x** |
| **FPS** | 8.2 | 25.0 | **3.0x** |
| **Accuracy** | 70% | 70% | Same |

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/baalaganeshr/Arm-Bharat-ai--challenge-.git
cd Arm-Bharat-ai--challenge-

# Run with Docker
docker compose up -d
docker exec -it mask-detection-dev bash
python modified/detect_cpu.py

# Or install locally
pip install -r requirements.txt
python modified/detect_cpu.py
```

---

## 📁 Project Structure

```
├── modified/          # Detection scripts
├── our_improvements/  # Dashboard & FPGA interface
├── models/            # Trained models (Git LFS)
├── docs/              # Full documentation
└── README.md          # This file
```

---

## 📚 Full Documentation

See [docs/README.md](docs/README.md) for complete documentation including:
- Detailed architecture
- Installation guide
- FPGA integration
- API reference

---

## 🙏 Acknowledgments

- Base code: [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)
- Bharat AI-SoC Challenge 2026

---

**Made with ❤️ by Team PSR**
