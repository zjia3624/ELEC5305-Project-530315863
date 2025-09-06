Speech Command Recognition in MATLAB (Bark-Spectrogram CNN)

Research Question： 
On clean Speech Commands, can a compact CNN trained on Bark spectrograms reach near-baseline accuracy while staying lightweight? 
We will also test a tiny variant (channels × 0.5) or replace Bark with Mel to study the accuracy–size trade-off.

Methodology：
MATLAB Online: Google Speech Commands. 
Bark spectrogram baseline, optional Mel. 
CNN per MathWorks example + tiny CNN. Speaker-stratified split.
minimal augmentation.

Evaluation： Accuracy, Macro-F1, Confusion Matrix. 
model Params and CPU latency (ms per 1-s clip).  
Planned figures. confusion matrix, accuracy-vs-model-size, learning curves.

Links：
Repo: https://github.com/zjia3624/ELEC5305-Project-zixianjia
Pages: https://zjia3624.github.io/ELEC5305-Project-zixianjia/
