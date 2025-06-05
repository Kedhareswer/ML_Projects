# 🧠 Machine Learning Projects Portfolio

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.4%2B-red.svg)](https://keras.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-green.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintained](https://img.shields.io/badge/Maintained-yes-brightgreen.svg)](https://github.com/Kedhareswer/ML_Projects/graphs/commit-activity)
[![Last Commit](https://img.shields.io/github/last-commit/Kedhareswer/ML_Projects)](https://github.com/Kedhareswer/ML_Projects/commits/main)

A comprehensive collection of machine learning projects showcasing various applications in computer vision, time series analysis, and predictive modeling.

## 📊 Projects Overview

| Project Name | Category | Technologies | Complexity | Status |
|-------------|----------|--------------|------------|---------|
| Emotion Analysis | Computer Vision | TensorFlow, Keras | ⭐⭐⭐⭐☆ | ✅ Complete |
| CIFAR-100 CNN | Image Classification | TensorFlow.js, CNN | ⭐⭐⭐⭐☆ | ✅ Complete |
| Forecasting Sticker Sales | Time Series | Prophet, Pandas | ⭐⭐⭐☆☆ | ✅ Complete |
| Video Game Sales EDA | Data Analysis | Pandas, Matplotlib | ⭐⭐☆☆☆ | ✅ Complete |
| House Price Prediction | Regression | Scikit-learn | ⭐⭐⭐☆☆ | ✅ Complete |
| Netflix Stock Price Prediction | Time Series | TensorFlow, LSTM | ⭐⭐⭐⭐☆ | ✅ Complete |
| MNIST | Image Classification | TensorFlow, Keras, TFLite | ⭐⭐⭐⭐☆ | ✅ Complete |
| Image to Sketch | Computer Vision / Image Processing | OpenCV, Pillow | ⭐⭐☆☆☆ | ✅ Complete |
| Moonknight | General | Python | ⭐⭐⭐☆☆ | ✅ Complete |
| Optimal Fertilizer Prediction | Predictive Modeling | Scikit-learn, Pandas | ⭐⭐⭐⭐☆ | ✅ Complete |
| Rainfall Prediction | Time Series | Pandas, Scikit-learn | ⭐⭐⭐⭐☆ | ✅ Complete |
| Sentiment Analysis (DistilBERT) | NLP | Transformers, Hugging Face | ⭐⭐⭐⭐☆ | ✅ Complete |

## 📈 Project Performance Metrics

```mermaid
graph LR
    A[ML Projects] --> B[Computer Vision]
    A --> C[Time Series]
    A --> D[Classification]
    B --> E[Emotion Analysis<br/>Accuracy: 85%]
    B --> F[CIFAR-100<br/>Accuracy: 78%]
    C --> G[Stock Prediction<br/>RMSE: 2.3]
    C --> H[Sales Forecast<br/>MAE: 1.8]
    D --> I[Image Classification<br/>F1: 0.82]
```

## 🌟 Featured Projects

### 1. Emotion Analysis 😊
<details>
<summary>Click to expand</summary>

#### Overview
Deep learning model for real-time emotion detection in images using CNN architecture.

#### Model Architecture
```
Input → Conv2D → BatchNorm → MaxPool → Conv2D → BatchNorm → MaxPool → Dense → Output
```

#### Performance
| Metric | Score |
|--------|-------|
| Accuracy | 85.2% |
| F1-Score | 0.83 |
| Training Time | 3464s |

#### Key Features
- Real-time emotion detection
- 7 emotion classes
- Data augmentation
- Batch normalization
</details>

### 2. CIFAR-100 CNN 🖼️
<details>
<summary>Click to expand</summary>

#### Overview
Convolutional Neural Network for classifying images from the CIFAR-100 dataset.

#### Implementation
- TensorFlow.js integration
- Custom CNN architecture
- Browser-based inference
- Real-time classification

#### Features
- 100 class classification
- Mobile-friendly design
- Optimized for web deployment
- Interactive visualization
</details>

### 3. Forecasting Sticker Sales 📊
<details>
<summary>Click to expand</summary>

#### Overview
Time series forecasting model for predicting sticker sales using Prophet.

#### Metrics
| Metric | Value |
|--------|--------|
| MAE | 1.8 |
| RMSE | 2.4 |
| R² | 0.86 |

#### Features
- Seasonal decomposition
- Trend analysis
- Future predictions
- Confidence intervals
</details>

### 4. MNIST Handwritten Digit Recognition ✍️
<details>
<summary>Click to expand</summary>

#### Overview
CNN model for classifying handwritten digits (0-9) from the MNIST dataset. This project demonstrates a robust pipeline including data augmentation, model training with various callbacks for optimal performance, detailed evaluation, and conversion of the trained model to TFLite format for deployment on edge devices.

#### Model Architecture
The model is a Sequential Convolutional Neural Network with the following key layers:
- Input layer with GaussianNoise for regularization.
- Two blocks of:
    - `Conv2D` (32 filters, then 64 filters) with `relu` activation and L2 regularization.
    - `BatchNormalization`.
    - `MaxPooling2D`.
    - `Dropout`.
- `GlobalAveragePooling2D` to reduce dimensions.
- A `Dense` hidden layer (128 units, `relu`, L2 regularization) with `BatchNormalization` and `Dropout`.
- A `Dense` output layer (10 units, `softmax` activation) for classification.

#### Performance
| Metric                          | Score   |
|---------------------------------|---------|
| Final Test Accuracy             | 99.55%  |
| Accuracy on Noisy Test Set    | 99.56%  |

#### Key Features
- High accuracy (over 99.5%) in handwritten digit classification.
- Effective data augmentation (rotation, width/height shifts, zoom, shear).
- Utilization of callbacks: `ModelCheckpoint` (save best), `EarlyStopping` (prevent overfitting), `ReduceLROnPlateau` (adaptive learning rate).
- Trained model saved in Keras format (`mnist_final_model.keras`).
- Model successfully converted to TensorFlow Lite format (`mnist_model.tflite`) for efficient deployment on mobile and edge devices.
- Comprehensive evaluation including a classification report and confusion matrix.
- Demonstrated robustness with high accuracy maintained on a noisy version of the test set.
</details>

## 💻 Technologies Used

```
Python        ████████████████████  100%
TensorFlow    ██████████████████    90%
Keras         ████████████████      80%
Scikit-learn  ████████████          60%
Prophet       ██████████            50%
Pandas        ████████████████      80%
Matplotlib    ████████████          60%
```

## 📋 Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥3.8 | Core Programming |
| TensorFlow | ≥2.0 | Deep Learning |
| Keras | ≥2.4 | Neural Networks |
| Scikit-learn | ≥1.0 | Machine Learning |
| Prophet | ≥1.0 | Time Series |
| Pandas | ≥1.3 | Data Management |
| Matplotlib | ≥3.4 | Visualization |

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kedhareswer/ML_Projects.git
   ```

2. **Set up the environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .nv\Scriptsctivate
   pip install -r requirements.txt
   ```

3. **Choose a project:**
   ```bash
   cd project_name
   jupyter notebook
   ```

## 📊 Project Comparison

| Project | Input Type | Output Type | Model Type | Accuracy |
|---------|------------|-------------|------------|-----------|
| Emotion Analysis | Images | 7 Emotions | CNN | 85.2% |
| CIFAR-100 | Images | 100 Classes | CNN | 78.3% |
| Sales Forecast | Time Series | Predictions | Prophet | 86% R² |
| Video Game Sales | Tabular | Analysis | EDA | N/A |
| House Prices | Tabular | Regression | Random Forest | 91.2% |
| Stock Prediction | Time Series | Predictions | LSTM | 87.5% |
| MNIST | Images (28x28 grayscale) | Digit (0-9) | CNN | 99.5% |
| Image to Sketch | Image | Image (sketch) | Image Processing | N/A |
| Moonknight | Text | Analysis/Classification | General ML | TBD |
| Optimal Fertilizer Prediction | Tabular Data | Fertilizer Recommendation | Predictive Model | TBD |
| Rainfall Prediction | Time Series Data | Rainfall Forecast | Time Series Model | TBD |
| Sentiment Analysis (DistilBERT) | Text | Sentiment Classification | Transformer (DistilBERT) | TBD |

## 📈 Project Stats

- **Total Projects**: 12
- **Technologies Used**: 7
- **Last Updated**: June 2025
- **Active Development**: Yes

## ⚠️ Important Notes

- Educational Purpose Only
- Datasets Not Included
- Models Require Training
- GPU Recommended for CNN Projects

## 📫 Contact & Support

- **Email**: Kedhareswer.12110626@gmail.com
- **Issues**: [GitHub Issues](https://github.com/Kedhareswer/ML_Projects/issues)
- **Discussion**: [GitHub Discussions](https://github.com/Kedhareswer/ML_Projects/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🌟 Star this repository if you find it helpful!

[![GitHub stars](https://img.shields.io/github/stars/Kedhareswer/ML_Projects.svg?style=social&label=Star)](https://github.com/Kedhareswer/ML_Projects)
[![GitHub forks](https://img.shields.io/github/forks/Kedhareswer/ML_Projects.svg?style=social&label=Fork)](https://github.com/Kedhareswer/ML_Projects/fork)

</div>

## AI Projects Portfolio

This section outlines several AI project ideas, complete with foundational structures for further development.

### [Ai Art Style Transfer](./ai_art_style_transfer/README.md)
- **Description:** AI Art Style Transfer App
- **Type:** DL (CV + GANs)
- **Value Summary:** Creative industry, AR filters, photo apps

### [Speech Emotion Recognition](./speech_emotion_recognition/README.md)
- **Description:** Real-Time Speech Emotion Recognition
- **Type:** Audio ML + DL
- **Value Summary:** Sentiment monitoring in support teams

### [Ai Game Bot](./ai_game_bot/README.md)
- **Description:** AI Game Bot (Deep Reinforcement Learning)
- **Type:** RL
- **Value Summary:** Entertainment, teaching reinforcement learning

### [Deepfake Voice Cloner](./deepfake_voice_cloner/README.md)
- **Description:** Real-Time Deepfake Voice Cloner
- **Type:** Audio DL (Voice Synthesis)
- **Value Summary:** Entertainment, dubbing, localization

### [Childrens Story Generator](./childrens_story_generator/README.md)
- **Description:** Children’s Story Generator with Illustrations
- **Type:** NLP + Image Gen
- **Value Summary:** Edutainment for kids and educators

### [Accent Converter](./accent_converter/README.md)
- **Description:** Accent Converter (Speech-to-Speech)
- **Type:** Audio ML + TTS
- **Value Summary:** Localization, language learning
