

# ELEC5305 Project – Noise-Robust Digit Speech Recognition

**Author:** Zixian Jia (SID: 530315863)  
**Unit:** ELEC5305 – Acoustics, Speech and Signal Processing  
**Date:** 16 / 11 / 2025  

---

## 1. Project Overview

This project studies **small-sample digit speech recognition** on the Free Spoken Digit Dataset (FSDD).  
Two systems are compared:

- **MFCC + SVM** (traditional feature + shallow model)  
- **Log-Mel spectrogram + CNN** (time–frequency feature + deep model)

We focus on **recognition accuracy**, **noise robustness** under different SNR levels (0–20 dB), and **inference latency**, aiming at **lightweight speech recognition for embedded devices**.

---

## 2. Files and Code

- 🔗 **GitHub Repository (Code)**  
[(https://github.com/zjia3624/ELEC5305-Project-530315863/tree/main/Code)](https://github.com/zjia3624/ELEC5305-Project-530315863/tree/main/Code)

-  **Project Figure**  
[(https://github.com/zjia3624/ELEC5305-Project-530315863/tree/main/Figures)](https://github.com/zjia3624/ELEC5305-Project-530315863/tree/main/Figures)

- 🎥 **Demo Video**  
  [Watch the project demo](在这里放你的 YouTube / Drive / OneDrive 链接)

---

## 3. Methodology (Short Summary)

### 3.1 Dataset

- Dataset: **Free Spoken Digit Dataset (FSDD)**  
- Digits 0–9, multiple speakers, 8 kHz sampling rate  
- Train / Test split: **80% / 20%**  
- For noise experiments, **white noise (AWGN)** is added only to the **test set** at SNR = 0, 5, 10, 15, 20 dB.

### 3.2 Feature Extraction

- **MFCC + Δ + ΔΔ (39-D)**  
  - 25 ms frame length, 10 ms frame shift  
  - 26 Mel filters, 13 static MFCC + first and second derivatives  
  - Flattened into fixed-length vectors → SVM classifier.

- **Log-Mel Spectrogram (40 × ~80)**  
  - STFT → Mel filter bank (40 filters) → log operation  
  - 2D time–frequency representation → input to CNN.

### 3.3 Models

- **SVM (MFCC input)**  
  - RBF kernel, small grid search over C and kernelScale  
  - Works on low-dimensional static features, prone to overfitting.

- **CNN (Log-Mel input)**  
  - 2 convolution + pooling layers  
  - 1 fully connected layer + softmax  
  - Trained for 20–30 epochs with Adam and cross-entropy loss.

---

## 4. Key Results

### 4.1 Clean Speech

- **SVM (MFCC)**  
  - High training accuracy, but only **57.33%** on the test set due to overfitting.

- **CNN (Log-Mel)**  
  - Final test accuracy **74.00%**, with stable training and validation curves.

👉 **Conclusion:** On clean FSDD, CNN (Log-Mel) significantly outperforms SVM (MFCC).

### 4.2 Noise Robustness (SNR = 0–20 dB)

- **SVM (MFCC)**  
  - Accuracy stays around **10%** for all SNR levels (almost random guess).

- **CNN (Log-Mel)**  
  - 0 dB: 16.67%  
  - 5 dB: ~20%  
  - 10 dB: ~31%  
  - 15 dB: ~46%  
  - 20 dB: ~61%

👉 **Conclusion:** CNN maintains clearly higher accuracy and shows a **monotonic improvement** as SNR increases, indicating much stronger **noise robustness**.

### 4.3 Inference Latency

- Average inference time of CNN: **0.0282 ms per sample** (20 runs averaged).  
- This meets real-time requirements for small speech recognition systems.

---

## 5. Discussion and Conclusion 

- **MFCC + SVM** suffers from:
  - Loss of detailed time–frequency structure due to strong compression.  
  - Inability to model temporal dynamics of speech.  
  - Extremely poor robustness to noise and unseen speakers.

- **Log-Mel + CNN** benefits from:
  - Full 2D time–frequency structure as input.  
  - Convolution kernels capturing local patterns (formants, energy peaks, transitions).  
  - Better generalization across speakers and SNR levels.  
  - Very low inference latency suitable for embedded deployment.

Overall, the experiments show that **Log-Mel + CNN** is a more effective and practical solution for small-sample digit speech recognition under noisy conditions, while **MFCC + SVM** is more suitable as a baseline method.

