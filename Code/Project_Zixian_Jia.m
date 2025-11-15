clear; clc; close all;
p = gcp('nocreate');
if ~isempty(p)
    delete(p);
end
warning('off','all');
rng(0, 'twister');

feature('DefaultCharacterSet','UTF-8');
trainingPlot.Language = "en";

%% 1. Configuration 

datasetRoot = fullfile(pwd, "recordings");   

fsTarget    = 8000;      % resample to 8 kHz
numMFCC     = 13;        % MFCC order
useDelta    = true;      % add delta & delta-delta

numMelBands = 40;        % log-mel bands for CNN
frameLen    = 256;       % STFT frame length
hopLen      = 128;       % STFT hop
maxTimeSec  = 1.0;       % pad/trim to 1 s

trainRatio  = 0.7;       % 70% train, 15% val, 15% test
valRatio    = 0.15;

snrLevels = [0 5 10 15 20]; % SNRs for robustness test (dB)

%% 2. Load dataset 
ads = audioDatastore(datasetRoot, ...          % create datastore that indexes all audio files
    "IncludeSubfolders", true, ...             % also search all subfolders under datasetRoot
    "FileExtensions", ".wav", ...              % only include .wav files
    "LabelSource", "none");                   % do not auto-generate labels (we set them manually)

numFiles = numel(ads.Files);                 % total number of audio files found

if numFiles == 0                           % sanity check: no files -> stop with error
    error("No wav files under %s", datasetRoot); % tell user that dataset folder is empty / wrong
end

% labels from filename first char
labels = cell(numFiles,1);                    % pre-allocate cell array for string labels

for i = 1:numFiles                              % loop over all files
    [~, name, ~] = fileparts(ads.Files{i});     % extract filename without path and extension
    digitChar = name(1);                        % assume first character is the digit label (0–9)
    if ~ismember(digitChar, '0123456789')       
        error("File '%s' does not start with digit 0-9.", name);  % throw descriptive error
    end
    labels{i} = digitChar;                       % store digit label as a char in the cell array
end

ads.Labels = categorical(labels);                % convert char labels to categorical class labels

fprintf("Loaded %d files.\n", numFiles);         % print total number of files
disp("Label distribution:");                     % show how many samples per digit
disp(countcats(ads.Labels));                     % count occurrences for each categorical label

%% 3. Train/val/test split 

[adsTrain, adsRest] = splitEachLabel(ads, trainRatio);        % split into training set and remaining data (val+test) with balanced labels
valPortion = valRatio / (1 - trainRatio);                     % convert global val ratio into ratio within the remaining portion
[adsVal, adsTest]  = splitEachLabel(adsRest, valPortion);     % split remaining data into validation and test sets (also label-balanced)

fprintf("Train: %d, Val: %d, Test: %d\n", ...                  % print number of files in each subset for sanity check
    numel(adsTrain.Files), numel(adsVal.Files), numel(adsTest.Files));

%% 4. Baseline features: MFCC(+delta) for SVM 

fprintf("Extracting MFCC-based features (baseline)...\n");
[XTrain_mfcc, yTrain] = extractFrameFeatures(adsTrain, fsTarget, ...
    "mfcc", numMFCC, useDelta, [], frameLen, hopLen, maxTimeSec);

[XVal_mfcc,   yVal]   = extractFrameFeatures(adsVal, fsTarget, ...
    "mfcc", numMFCC, useDelta, [], frameLen, hopLen, maxTimeSec);

[XTest_mfcc,  yTest]  = extractFrameFeatures(adsTest, fsTarget, ...
    "mfcc", numMFCC, useDelta, [], frameLen, hopLen, maxTimeSec);

%% 5. Baseline model: SVM with small grid search 

params.kernelScaleList   = [0.5 1 2];                         % candidate values for RBF kernel scale
params.boxConstraintList = [0.5 1 2];                         % candidate values for SVM box constraint C

fprintf("Training baseline SVM (MFCC features)...\n");        
[svmModel, svmCfg] = trainSVMWithValidation( ...              % train SVM on MFCC(+delta) features and select best hyper-parameters
    XTrain_mfcc, yTrain, params);

fprintf("Best SVM: kernelScale=%.2f, C=%.2f\n", ...           % report chosen hyper-parameters from validation
    svmCfg.kernelScale, svmCfg.boxConstraint);

[~, accTrain_svm] = evaluateClassifier(svmModel, ...          % accuracy on training set (overfitting indicator)
    XTrain_mfcc, yTrain);
[~, accVal_svm]   = evaluateClassifier(svmModel, ...          % accuracy on validation set (model selection)
    XVal_mfcc,   yVal);
[yPred_svm, accTest_svm, confTest_svm] = evaluateClassifier( ...  % accuracy and confusion matrix on held-out test set
    svmModel, XTest_mfcc, yTest);

fprintf("Baseline SVM Acc: train=%.2f%%, val=%.2f%%, test=%.2f%%\n", ...  % summary of SVM performance
    accTrain_svm*100, accVal_svm*100, accTest_svm*100);

figure;
confusionchart(confTest_svm, categories(yTest));              % visualize per-class performance using confusion matrix
xlabel("Predicted Class");
ylabel("True Class");
title(sprintf("Baseline SVM (MFCC) - Test Acc %.2f%%", ...    % title includes overall test accuracy
    accTest_svm*100));

%% 6. CNN features: log-mel spectrogram "images"

fprintf("Extracting log-mel spectrograms for CNN...\n");
[XTrain_cnn, yTrain_cnn] = extractSpectrogramSet(adsTrain, fsTarget, ...
    numMelBands, frameLen, hopLen, maxTimeSec, []);

[XVal_cnn,   yVal_cnn]   = extractSpectrogramSet(adsVal, fsTarget, ...
    numMelBands, frameLen, hopLen, maxTimeSec, []);

[XTest_cnn,  yTest_cnn]  = extractSpectrogramSet(adsTest, fsTarget, ...
    numMelBands, frameLen, hopLen, maxTimeSec, []);

inputSize = size(XTrain_cnn(:,:,:,1));  

%% 7. Define and train small CNN 

layers = buildCnnLayers(inputSize, numel(categories(yTrain_cnn)));  % create a small CNN architecture for log-mel inputs

miniBatchSize = 64;                                                 % number of samples per mini-batch
maxEpochs     = 20;                                                % maximum number of training epochs

opts = trainingOptions("adam", ...                                  % use Adam optimizer for training
    "MaxEpochs", maxEpochs, ...                                    % total number of epochs
    "MiniBatchSize", miniBatchSize, ...                              % mini-batch size
    "Shuffle", "every-epoch", ...                                   % shuffle training data at the start of each epoch
    "ValidationData", {XVal_cnn, yVal_cnn}, ...                      % validation set used to monitor generalisation
    "ValidationFrequency", 30, ...                                  % validate every 30 iterations
    "Verbose", true, ...                                            % print training progress to command window
    "Plots", "training-progress");                                 % show live training/validation curves

fprintf("Training CNN (log-mel)...\n");                             % message to indicate that CNN training starts
net = trainNetwork(XTrain_cnn, yTrain_cnn, layers, opts);           % train the CNN on log-mel spectrogram inputs

%% 8. Evaluate CNN on clean test set + latency 

% 8.1 accuracy
tStart = tic;                                                        % start timer for single forward pass
yPred_cnn_clean = classify(net, XTest_cnn);                          % classify all clean test samples with trained CNN
tElapsed = toc(tStart);                                              % total time for one pass over the test set

accTest_cnn = mean(yPred_cnn_clean == yTest_cnn);                    % overall test accuracy on clean data
fprintf("CNN Test Acc (clean): %.2f%%\n", accTest_cnn*100);          % print accuracy as percentage
fprintf("Single pass over %d test samples took %.4f s (%.4f s / sample)\n", ... % timing information
    numel(yTest_cnn), tElapsed, tElapsed/numel(yTest_cnn));

% To obtain a more stable delay estimate, multiple repeated measurements are performed.
numRepeats = 20;                                                     % number of repeated timing runs
latArray = zeros(numRepeats,1);                                      % store latency of each run
for r = 1:numRepeats                                                 % repeat forward pass several times
    t0 = tic;                                                        % start timer for this run
    classify(net, XTest_cnn);                                       % run classification (results are ignored here)
    latArray(r) = toc(t0);                                           % store elapsed time for this run
end
avgLatencyPerSample = mean(latArray) / numel(yTest_cnn);            % average latency per sample over all runs
fprintf("Average CNN latency per sample over %d runs: %.4f ms\n", ...% print mean latency in milliseconds
    numRepeats, avgLatencyPerSample*1000);

% 8.2 Confusion Matrix
figure;
confTest_cnn_clean = confusionmat(yTest_cnn, yPred_cnn_clean);     % compute confusion matrix for clean test data
confusionchart(confTest_cnn_clean, categories(yTest_cnn));
xlabel("Predicted Class");
ylabel("True Class");
title(sprintf("CNN (Log-Mel) - Test Acc %.2f%%%%", accTest_cnn*100));

%% 9. Noise robustness for both SVM and CNN

accNoisy_svm      = zeros(size(snrLevels));                     % SVM accuracy for each SNR level
accNoisy_cnn      = zeros(size(snrLevels));                     % CNN accuracy for each SNR level
latencyNoisy_cnn  = zeros(size(snrLevels));                     % CNN average latency per sample for each SNR

confNoisy_svm     = cell(size(snrLevels));                      % store SVM confusion matrix at each SNR
confNoisy_cnn     = cell(size(snrLevels));                      % store CNN confusion matrix at each SNR

for k = 1:numel(snrLevels)                                     % loop over all SNR conditions
    snr = snrLevels(k);                                        % current SNR value in dB
    fprintf("Noise test: SNR = %d dB", snr);        % header for this noise condition

% 9.1 SVM baseline: MFCC(+delta)
    [XTest_mfcc_noisy, yTest_noisy] = extractFrameFeatures( ...
        adsTest, fsTarget, ...
        "mfcc", numMFCC, useDelta, snr, ...
        frameLen, hopLen, maxTimeSec);

    [yPred_svm_noisy, accNoisy_svm(k), confNoisy_svm{k}] = ...
        evaluateClassifier(svmModel, XTest_mfcc_noisy, yTest_noisy);

    fprintf("  SVM (MFCC) accuracy %2d dB = %.2f%%%%\n", ...
        snr, accNoisy_svm(k)*100);

% 9.2 CNN: log-mel
    [XTest_cnn_noisy, yTest_cnn_noisy] = extractSpectrogramSet( ...
        adsTest, fsTarget, ...
        numMelBands, frameLen, hopLen, maxTimeSec, snr);

    tStart = tic;
    yPred_cnn_noisy = classify(net, XTest_cnn_noisy);
    tElapsed = toc(tStart);

   accNoisy_cnn(k)     = mean(yPred_cnn_noisy == yTest_cnn_noisy);      % CNN accuracy at this SNR
    confNoisy_cnn{k}    = confusionmat(yTest_cnn_noisy, yPred_cnn_noisy);      % confusion matrix for CNN
    latencyNoisy_cnn(k) = tElapsed / numel(yTest_cnn_noisy);        % average latency per sample (seconds/sample)
    fprintf("  CNN (log-mel) accuracy %2d dB = %.2f%%%%, latency = %.3f ms/sample\n", ...
        snr, accNoisy_cnn(k)*100, latencyNoisy_cnn(k)*1000);
end

%% 10. Plot SNR vs accuracy (SVM vs CNN) 


snrAxisAcc = [snrLevels 100];    
accSvmAll  = [accNoisy_svm accTest_svm];
accCnnAll  = [accNoisy_cnn accTest_cnn];

figure;
plot(snrAxisAcc, accSvmAll*100, "-o", "LineWidth",1.5); hold on;
plot(snrAxisAcc, accCnnAll*100, "-s", "LineWidth",1.5);
grid on;

xticks(snrAxisAcc);
xticklabels([string(snrLevels)+" dB","clean"]);

xlabel("SNR");
ylabel("Accuracy (%)");
legend("SVM (MFCC)","CNN (Log-Mel)","Location","southwest");
title("Noise robustness comparison: SVM vs CNN");
hold off;

%% 11. Plot SNR vs CNN latency 

figure;
snrAxis = double(snrLevels(:));     
lat_ms   = latencyNoisy_cnn(:) * 1000;

plot(snrAxis, lat_ms, "-o", "LineWidth", 1.8, "MarkerSize", 6);
grid on;

xlabel("SNR (dB)");
ylabel("Avg CNN latency per sample (ms)");
title("CNN inference latency vs SNR");


set(gca,"XTick",snrAxis);
set(gca,"XTickLabel", string(snrAxis) + " dB");


%% 12. Plot robustness comparison 

figure;
snrPlot = [Inf snrLevels];
plot(snrPlot, [accTest_svm accNoisy_svm], "-o", ...
     snrPlot, [accTest_cnn accNoisy_cnn], "-s", "LineWidth", 1.5);
xticks([0 10 20 Inf]);
xticklabels({"0","10","20","clean"});
xlabel("SNR (dB, 'clean' = Inf)");
ylabel("Accuracy");
legend("SVM (MFCC)", "CNN (Log-Mel)", "Location", "southwest");
grid on;
title("Noise robustness comparison: SVM vs CNN");

%% 13. Save everything for report 

results.fsTarget     = fsTarget;
results.accTrain_svm = accTrain_svm;
results.accVal_svm   = accVal_svm;
results.accTest_svm  = accTest_svm;
results.accTest_cnn  = accTest_cnn;
results.snrLevels    = snrLevels;
results.accNoisy_svm = accNoisy_svm;
results.accNoisy_cnn = accNoisy_cnn;
results.svmCfg       = svmCfg;

save("spoken_digit_advanced_results.mat", "-struct", "results");
fprintf("Saved results to spoken_digit_advanced_results.mat\n");