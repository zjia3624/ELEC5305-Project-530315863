Voice command recognition using deep learning



Name: zixian jia
Sid: 530315863
GitHub Username: zjia3624



1. Overview

This project targets end-devices such as headphones, watches, and smart homes. It aims to build a small-vocabulary speech recognition (KWS) system that can run in real time on a CPU. 

The main obstacles to current deployment are: 1. Large model size and computational complexity, resulting in high latency and energy consumption. 
2. Insufficient recognition robustness under environmental noise and gain variations.

Therefore, we proposed using PCEN-Mel as the front-end feature and designed two lightweight network routes. Depthwise separable convolution DS-CNN and one-dimensional temporal convolution TCN adjusting the number of channels using the width coefficient. we systematically evaluated the trade-off between accuracy, parameter count, and CPU latency.

2. Background

In recent years, deep learning has achieved high accuracy in KWS, but many methods rely on deep 2D CNNs or attention structures, which are not suitable for mobile deployment. Furthermore, traditional MFCC/Log Mel performance degrades significantly under conditions of gain drift or distant speech. PCEN, on the other hand, utilizes channel-level adaptive gain suppression and dynamic compression, making it more robust in low SNR and distant speech scenarios. Depthwise separable convolution can achieve comparable accuracy with very few parameters, and TCN models long-term dependencies with low latency through dilated convolution. The reason for choosing this project is that it has both research value and engineering significance and is predicted to be completed within a limited time.


3. Methodology

This project conducts experiments based on Matlab and Google Speech Commands. PCEN-Mel features are extracted from speech clips and enhanced using SpecAugment and a noise mixture. Two lightweight networks are trained, the data is divided into layers by speaker, and learning curves are plotted.





Links：
Repo: https://github.com/zjia3624/ELEC5305-Project-530315863

