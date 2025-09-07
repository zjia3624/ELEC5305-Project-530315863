Voice command recognition using deep learning

Name: zixian jia
Sid: 530315863
GitHub Username: zjia3624


Overview

This project aims to build a small-vocabulary voice command recognition system that can run in real time on a CPU, targeting edge devices such as headphones, watches, and home devices. Moderate data augmentation is used to improve stability in everyday noise environments.

Background

Although existing KWS deep models are highly accurate, they are often difficult to deploy in real time on mobile devices due to their large size and lack of noise robustness. Therefore, this project uses PCEN-Mel features and lightweight DS-CNN/TCN to explore a voice command recognition solution that balances robustness and low latency on embedded devices.



Methodology

This project conducts experiments based on Matlab and Google Speech Commands. PCEN-Mel features are extracted from speech clips and enhanced using SpecAugment and a noise mixture. Two lightweight networks are trained, the data is divided into layers by speaker, and learning curves are plotted.





Links：
Repo: https://github.com/zjia3624/ELEC5305-Project-530315863

