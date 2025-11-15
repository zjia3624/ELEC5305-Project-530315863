# Code folder
## Code Structure and File Descriptions

All MATLAB scripts used in this project are located in the repository root.  
This section explains the purpose of each file so that reviewers can easily understand the workflow and reproduce the experiments.

---

### Main Scripts

- **`Project_Zixian_Jia.m`**  
  The main script (standard `.m` format) that runs the entire experimental pipeline:
  1. Load spoken digit audio data from the `recordings/` folder  
  2. Split into training / validation / test sets  
  3. Extract MFCC(+Δ+ΔΔ) features and train the SVM baseline  
  4. Extract log-Mel spectrograms and train the CNN  
  5. Evaluate SVM and CNN performance on clean and noisy test sets  
  6. Measure CNN inference latency  
  7. Save all results to `spoken_digit_advanced_results.mat`

- **`project.mlx`**  
  A MATLAB Live Script version of the experiment (same logic as above).  
  It contains formatted text, inline output, and plots—recommended for interactive viewing.

---

### Feature Extraction & Model Functions

- **`addNoiseToSignal.m`**  
  Function: `y = addNoiseToSignal(x, targetSNRdB)`  
  Adds Gaussian white noise to a clean signal based on a specified SNR level.  
  Used during noise robustness testing.

- **`buildCnnLayers.m`**  
  Function: `layers = buildCnnLayers(inputSize, numClasses)`  
  Constructs a 2D CNN architecture for log-Mel spectrogram input.  
  Includes conv–BN–ReLU–pool blocks, dropout, fully connected layers, and softmax output.

- **`evaluateClassifier.m`**  
  Function: `[yPred, acc, confMat] = evaluateClassifier(model, X, yTrue)`  
  Makes predictions, computes accuracy, and returns a confusion matrix.  
  Works for both SVM and CNN classifiers.

- **`extractFrameFeatures.m`**  
  Function:  
  ```matlab
  [X, y] = extractFrameFeatures(ads, fsTarget, featType, ...
                                numCoeffs, useDelta, snrTarget, ...
                                frameLen, hopLen, maxTimeSec)
