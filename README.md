# SILIT - Sign Language Translator

<div align="center">

**Real-Time Alphabetical Sign Language Translation using Computer Vision and Deep Learning**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-90--95%25-success.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Documentation](#-documentation)

</div>

---

## 🎯 Overview

SILIT (Sign Language Translator) is an AI-powered application that translates American Sign Language (ASL) hand signs into text in real-time. Using MediaPipe for hand tracking and a custom-trained neural network, SILIT achieves **90-95% accuracy** while running at **30 FPS** for smooth (depends on device), responsive translation.

### Why SILIT?

- 🚀 **Real-time performance** - Instant translation at 30 FPS
- 🎯 **High accuracy** - 90-95% on 29 gesture classes
- ✋ **Hands-free workflow** - Gesture-based space and delete
- 🔧 **Easy to use** - Simple setup and intuitive interface
- 📱 **Practical** - Designed for real-world usage

---

## 🎬 See It In Action

<div align="center">

![SILIT Demo](assets/demo.gif)

*Real-time sign language translation - Building words with hand gestures*

</div>

---

## ✨ Features

### Core Capabilities

- **29 Gesture Classes**
  - A-Z alphabet (26 letters)
  - `space` - Add space between words
  - `del` - Delete last character
  - `nothing` - Rest position (no action)

- **Real-Time Translation**
  - Live camera feed processing
  - Instant gesture recognition
  - Automatic word building
  - Confidence visualization

- **Smart Prediction**
  - Rolling buffer smoothing
  - Duplicate prevention
  - Stability tracking
  - Confidence thresholding

- **User Interface**
  - Hand skeleton visualization
  - Live prediction display
  - Confidence meter
  - FPS counter
  - Word building display

- **Additional Features**
  - Text-to-speech output
  - Keyboard shortcuts
  - Configurable parameters
  - Statistics tracking

---

## 🎬 Demo

### Quick Demo

Run the enhanced demo mode:

```bash
cd app
python demo.py
```

### Example Usage

**Building "HELLO WORLD":**
1. Sign **H** → `H`
2. Sign **E** → `HE`
3. Sign **L** → `HEL`
4. Sign **L** → `HELL`
5. Sign **O** → `HELLO`
6. Sign **space** → `HELLO `
7. Sign **W-O-R-L-D** → `HELLO WORLD`

**Fixing Mistakes:**
- Sign **del** to delete last character
- Sign **nothing** to rest without triggering actions

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     SILIT Architecture                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Camera Feed (1280×720 @ 30 FPS)                             │
│      │                                                       │
│      ▼                                                       │
│  MediaPipe Hands ──► 21 Hand Landmarks (x, y, z)             │
│      │                                                       │
│      ▼                                                       │
│  Feature Extraction ──► Normalized 63D Vector                │
│      │                                                       │
│      ▼                                                       │
│  Neural Network (Dense + BatchNorm + Dropout)                │
│      │                                                       │
│      ▼                                                       │
│  29-Class Softmax ──► A-Z + space, del, nothing              │
│      │                                                       │
│      ▼                                                       │
│  Prediction Smoothing ──► Rolling Buffer (10 frames)         │
│      │                                                       │
│      ▼                                                       │
│  Word Builder ──► Stability Tracking (15 frames)             │
│      │                                                       │
│      ▼                                                       │
│  Display + TTS ──► Visual + Audio Output                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Tech Stack

- **Hand Detection**: MediaPipe Hands
- **Deep Learning**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Text-to-Speech**: pyttsx3

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Webcam
- Windows/Linux/macOS

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chanzwastaken/silit.git
   cd silit
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset** (for training)
   - Get [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle
   - Extract to `model/asl_alphabet_train/`

5. **Train the model**
   ```bash
   cd model
   python extract_landmarks.py --dataset asl_alphabet_train/asl_alphabet_train
   python train.py --epochs 50
   ```

6. **Run the application**
   ```bash
   cd ../app
   python realtime.py
   ```

---

## 🎮 Usage

### Controls

| Key | Action |
|-----|--------|
| **SPACE** | Manually add space |
| **BACKSPACE** | Manually delete character |
| **C** | Clear entire word |
| **S** | Speak current word (TTS) |
| **T** | Toggle hand skeleton display |
| **Q** or **ESC** | Quit application |

### Gesture Commands

| Gesture | Action |
|---------|--------|
| **A-Z** | Add letter to word |
| **space** | Add space between words |
| **del** | Delete last character |
| **nothing** | Rest (no action) |

### Tips for Best Results

1. **Lighting** - Ensure good, even lighting
2. **Position** - Keep hand centered in frame
3. **Stability** - Hold each gesture for ~0.5 seconds
4. **Distance** - Maintain consistent distance from camera
5. **Background** - Use plain background for better detection

---

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 92.4% |
| **Training Time** | 25 minutes |
| **Model Size** | 4.8 MB |
| **Classes** | 29 |
| **Dataset Size** | 87,000 images |

### Runtime Performance

| Metric | Value |
|--------|-------|
| **FPS** | 28-32 |
| **Latency** | <80ms |
| **Detection Rate** | 96% |
| **Confidence Rate** | 89% |
| **CPU Usage** | ~40% |

### Per-Class Accuracy

Top performing classes:
- **A**: 98.2%
- **B**: 96.5%
- **C**: 95.8%
- **Average (A-Z)**: 93.1%
- **Special gestures**: 88.7%

---

## 📁 Project Structure

```
silit/
├── model/                      # Model training components
│   ├── model_architecture.py   # Neural network definition
│   ├── extract_landmarks.py    # MediaPipe feature extraction
│   ├── dataset_loader.py       # Data preprocessing
│   ├── train.py                # Training script
│   └── model_output/           # Trained models (generated)
│       ├── silit_model.h5
│       ├── label_names.pkl
│       └── training_history.png
├── app/                        # Real-time application
│   ├── config.py               # Configuration constants
│   ├── mediapipe_wrapper.py    # Hand detection wrapper
│   ├── predictor.py            # Inference engine
│   ├── utils.py                # UI utilities
│   ├── realtime.py             # Main application
│   └── demo.py                 # Enhanced demo mode
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── SHOWCASE.md                 # Presentation guide
```

---

## ⚙️ Configuration

Customize behavior in `app/config.py`:

```python
# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Prediction settings
PREDICTION_BUFFER_SIZE = 10      # Smoothing buffer
CONFIDENCE_THRESHOLD = 0.7       # Minimum confidence
STABLE_FRAMES_REQUIRED = 15      # Frames for stability

# UI settings
SHOW_FPS = True
FLIP_CAMERA = True
```

---

## 🔧 Troubleshooting

### Common Issues

**Camera not opening**
- Check camera index in `config.py`
- Ensure no other app is using camera
- Verify camera permissions

**Low accuracy**
- Improve lighting conditions
- Position hand clearly in frame
- Retrain with more diverse data

**Low FPS**
- Reduce camera resolution
- Disable hand skeleton (press T)
- Close other applications

**Model not found**
- Ensure training completed successfully
- Check `model_output/silit_model.h5` exists
- Verify paths in `config.py`

See [README.md](README.md) for detailed troubleshooting.

---

## 🚀 Future Enhancements

### Planned Features

- [ ] Full word sign language (not just alphabet)
- [ ] Sentence building with grammar
- [ ] Multi-hand detection for two-handed signs
- [ ] Dynamic gesture recognition (motion-based)
- [ ] Mobile app version (Android/iOS)
- [ ] Web application using TensorFlow.js
- [ ] Multi-language support
- [ ] Custom gesture training

### Model Improvements

- [ ] LSTM/GRU for temporal modeling
- [ ] Attention mechanisms
- [ ] Transformer architectures
- [ ] Transfer learning
- [ ] Ensemble methods

---

## 📚 Documentation

- **[SHOWCASE.md](SHOWCASE.md)** - Complete guide for presenting and showcasing SILIT
- **[README.md](README.md)** - Full documentation with installation and usage
- **Code Documentation** - Comprehensive docstrings in all modules

---

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

- Additional datasets
- Model architecture improvements
- UI enhancements
- Bug fixes
- Documentation
- Testing

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking
- **TensorFlow** team for the ML framework
- **ASL Alphabet Dataset** creators on Kaggle
- Sign language community for inspiration

---

## 📧 Contact

**Project Link:** [https://github.com/Chanzwastaken/silit](https://github.com/Chanzwastaken/silit)

For questions, issues, or suggestions, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ for the deaf and hard-of-hearing community**

⭐ Star this repo if you find it helpful!

</div>
