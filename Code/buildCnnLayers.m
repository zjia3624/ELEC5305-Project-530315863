function layers = buildCnnLayers(inputSize, numClasses)
% Build a small 2D CNN for log-mel spectrogram input.

layers = [
    imageInputLayer(inputSize, "Name","input", "Normalization","zscore")

    convolution2dLayer(3, 16, "Padding","same", "Name","conv1")
    batchNormalizationLayer("Name","bn1")
    reluLayer("Name","relu1")
    maxPooling2dLayer([2 2], "Stride",2, "Name","pool1")

    convolution2dLayer(3, 32, "Padding","same", "Name","conv2")
    batchNormalizationLayer("Name","bn2")
    reluLayer("Name","relu2")
    maxPooling2dLayer([2 2], "Stride",2, "Name","pool2")

    convolution2dLayer(3, 64, "Padding","same", "Name","conv3")
    batchNormalizationLayer("Name","bn3")
    reluLayer("Name","relu3")
    maxPooling2dLayer([2 2], "Stride",2, "Name","pool3")

    dropoutLayer(0.3, "Name","drop1")

    fullyConnectedLayer(128, "Name","fc1")
    reluLayer("Name","relu4")
    dropoutLayer(0.3, "Name","drop2")

    fullyConnectedLayer(numClasses, "Name","fc_out")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")
];
end
