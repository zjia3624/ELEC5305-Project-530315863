# Code folder
## Code Structure and File Descriptions

All MATLAB scripts in this project are placed in the repository root directory.  
This section briefly explains the purpose of each file to help reviewers and instructors understand and reproduce the experiments.

---

### Main Scripts / Experiment Pipeline

- **`Project_Zixian_Jia.m`**  
  - The main script of the entire experiment (pure `.m` version).  
  - It performs the following steps:
    1. Load spoken digit audio data from the `recordings/` folder  
    2. Split data into training / validation / test sets (70% / 15% / 15%)  
    3. Extract MFCC(+Δ+ΔΔ) features and train the SVM baseline model  
    4. Extract log-Mel spectrograms for CNN input and train a small CNN  
    5. Evaluate SVM and CNN accuracy on both clean and noisy test sets (multiple SNR levels)  
    6. Measure CNN inference latency and plot the results  
    7. Save all important results into `spoken_digit_advanced_results.mat`  

- **`project.mlx`**  
  - A MATLAB Live Script version of the experiment with the same logic as `Project_Zixian_Jia.m`.  
  - Includes formatted sections, inline output, and visualizations.  
  - Recommended for interactive viewing of training logs, confusion matrices, and curves.

---

### Functional Files (Feature Extraction & Model Components)

- **`addNoiseToSignal.m`**  
  - Function: `y = addNoiseToSignal(x, targetSNRdB)`  
  - Purpose: Computes the required noise power for a given `targetSNRdB` and adds Gaussian white noise to signal `x`.  
  - Usage: Used for generating noisy test sets in noise robustness experiments at 0/5/10/15/20 dB SNR.

- **`buildCnnLayers.m`**  
  - Function: `layers = buildCnnLayers(inputSize, numClasses)`  
  - Builds a 2D CNN architecture for log-Mel spectrogram input.  
  - Main structure:
    - Input layer with z-score normalization  
    - Three convolutional blocks (`conv2d + batchNorm + ReLU + maxPooling`)  
    - Dropout layer  
    - Fully connected layer + Softmax + Classification layer  
  - Usage: Called by the main script during CNN training.

- **`evaluateClassifier.m`**  
  - Function: `[yPred, acc, confMat] = evaluateClassifier(model, X, yTrue)`  
  - Performs prediction on feature matrix `X`, returns classification accuracy `acc`, and the confusion matrix `confMat`.  
  - Usage: Used to evaluate both SVM and CNN models under different datasets or noise settings.

- **`extractFrameFeatures.m`**  
  - Function:
    ```matlab
    [X, y] = extractFrameFeatures(ads, fsTarget, featType, ...
                                  numCoeffs, useDelta, snrTarget, ...
                                  frameLen, hopLen, maxTimeSec)
    ```
  - Purpose: Reads audio files from `audioDatastore` and performs:
    - Resampling to `fsTarget`  
    - Time normalization using `fixLength`  
    - Optional noise addition (`snrTarget`)  
    - Frame-level MFCC extraction  
    - Mean pooling over time to form a fixed-length feature vector  
  - Usage: Feature extraction for the **MFCC + SVM baseline**.

- **`extractSpectrogramSet.m`**  
  - Function:
    ```matlab
    [X4d, y] = extractSpectrogramSet(ads, fsTarget, ...
                                     numMelBands, frameLen, hopLen, ...
                                     maxTimeSec, snrTarget)
    ```
  - Purpose: Extracts 4D log-Mel spectrogram “images” for CNN training:
    - Resampling + fixed length  
    - Optional noise injection  
    - Mel spectrogram → log10  
    - Output shape:  
      `numBands × numFrames × 1 × numFiles`  
  - Usage: Provides CNN input tensors, similar to image batches.

- **`fixLength.m`**  
  - Function: `xOut = fixLength(x, fs, maxTimeSec)`  
  - Purpose: Standardizes duration of all signals:
    - Trim if longer than target  
    - Zero-pad if shorter  
    - Leave unchanged if equal  
  - Usage: Ensures consistent duration for feature extraction and CNN batch training.

- **`trainSVMWithValidation.m`**  
  - Function:  
    ```matlab
    [bestModel, bestCfg] = trainSVMWithValidation(XTrain, yTrain, params)
    ```
  - Performs a grid search over SVM hyperparameters:
    - Loops through `kernelScaleList` and `boxConstraintList`  
    - Uses 5-fold cross-validation (`fitcecoc + kfoldLoss`)  
    - Prints validation accuracy for each configuration  
    - Selects the best performing model  
  - Usage: Provides an optimized **MFCC + SVM baseline model**.

---

### Results File

- **`spoken_digit_advanced_results.mat`**  
  - Contains a saved `results` structure including:
    - `fsTarget` — target sampling rate  
    - SVM accuracies on training / validation / test sets  
    - CNN test-set accuracy (clean condition)  
    - `snrLevels` — SNR list used for robustness experiments  
    - `accNoisy_svm`, `accNoisy_cnn` — SVM / CNN accuracy under different SNR levels  
  - Usage: Convenient for generating plots and tables without retraining models.

---

### How to Run the Code

1. **Check the dataset folder**

   Make sure the folder **`recordings/`** is present in the repository root and contains the spoken digit `.wav` files  
   (this repository already includes them, so normally no extra action is needed).

2. **Open the project in MATLAB**

   - Set the **Current Folder** in MATLAB to the repository root directory.
   - All `.m` files and the `recordings/` folder should be visible in the MATLAB file browser.

3. **Option 1 – Run the Live Script (recommended)**  

   - Open `project.mlx` in MATLAB.  
   - Execute the sections one by one (or “Run all”) to:
     - load the dataset,  
     - train the SVM baseline and the CNN,  
     - evaluate accuracy under clean and noisy conditions,  
     - display confusion matrices and plots,  
     - save the results to `spoken_digit_advanced_results.mat`.

4. **Option 2 – Run the plain script**

   - Run `Project_Zixian_Jia.m`.  
   - All steps (data loading, feature extraction, model training, evaluation and saving results) will execute in batch.  
   - Numerical results will be printed in the Command Window, and the figures (confusion matrices, SNR vs. accuracy, latency curve, etc.) will appear in separate figure windows.
