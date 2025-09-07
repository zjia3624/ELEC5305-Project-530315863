Voice command recognition using deep learning


Name: zixian jia

Sid: 530315863

GitHub Username: zjia3624

Links：
Repo: https://github.com/zjia3624/ELEC5305-Project-530315863


## Overview

This project targets end-devices such as headphones, watches, and smart homes. It aims to build a small-vocabulary speech recognition (KWS) system that can run in real time on a CPU. 

The main obstacles to current deployment are: 
Large model size and computational complexity, resulting in high latency and energy consumption. 
Insufficient recognition robustness under environmental noise and gain variations.

Therefore, we proposed using PCEN-Mel as the front-end feature and designed two lightweight network routes. Depthwise separable convolution DS-CNN and one-dimensional temporal convolution TCN adjusting the number of channels using the width coefficient. we systematically evaluated the trade-off between accuracy, parameter count, and CPU latency.


## Background

In recent years, deep learning has achieved high accuracy in KWS, but many methods rely on deep 2D CNNs or attention structures, which are not suitable for mobile deployment. Furthermore, traditional MFCC/Log Mel performance degrades significantly under conditions of gain drift or distant speech.
PCEN, on the other hand, utilizes channel-level adaptive gain suppression and dynamic compression, making it more robust in low SNR and distant speech scenarios. 

Depthwise separable convolution can achieve comparable accuracy with very few parameters, and TCN models long-term dependencies with low latency through dilated convolution. The reason for choosing this project is that it has both research value and engineering significance and is predicted to be completed within a limited time.



## Methodology

This project uses MATLAB and the 10 command types of Google Speech Commands v2 as the target. It divides the data into three categories: train/val/test according to the speaker.

The front-end uses PCEN-Mel features. SpecAugment and noise mixing are added during training to improve robustness. The model also uses two lightweight paths: DS-CNN and 1D-TCN. The number of channels is uniformly scaled using a width factor α. The Adam optimizer uses early stopping and a fixed random seed to ensure reproducibility. Accuracy, Macro-F1, and confusion matrix are used as accuracy metrics. Scatter plots are plotted and saved.



## Expected Outcomes

The project aims to produce a small-vocabulary KWS prototype that can run in real time on a CPU. Three core visualizations will be developed: confusion matrix, training/validation learning curves, and Accuracy vs. Model Size. The performance goal is to achieve an accuracy of ≥75% on a clean test set, and provide a brief error analysis for easily confused categories.


## Timeline

W6-W7:  Completed data cleaning and speaker stratification, PCEN-Mel feature caching, and the first round of training/validation closed loop, producing the initial confusion matrix and curves.

W8-W9:  Focusing on the main results, we prioritized training DS-CNN to form a scale curve and plotted the Accuracy-vs-Model-Size graph for the first time.

W10-W11: Conduct noise assessments and refine models.

W12-W13: Generate graphics styles and documentation, and improve the README and demo. Verify the process of reproducing the problem from scratch and submit a final report with a link.


## Reference

Wang, X., Sun, S., & Xie, L. (2019, December). Virtual adversarial training for DS-CNN based small-footprint keyword spotting. In 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) (pp. 607-612). IEEE.

Rusci, M., Van Hamme, H., & Tuytelaars, T. (2025, April). Self-Incremental Training for Personalized Voice Command Recognition in a Wireless Audio Sensor Network. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE.



Li, S., Yao, Y., Hu, J., Liu, G., Yao, X., & Hu, J. (2018). An ensemble stacked convolutional neural network model for environmental event sound recognition. Applied Sciences, 8(7), 1152.





