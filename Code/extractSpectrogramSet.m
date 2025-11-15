function [X4d, y] = extractSpectrogramSet(ads, fsTarget, ...
    numMelBands, frameLen, hopLen, maxTimeSec, snrTarget)

reset(ads);
numFiles = numel(ads.Files);
y = ads.Labels;

% Determine the spectrum size
[x0, fs0] = audioread(ads.Files{1});
x0 = mean(x0,2);
x0 = fixLength(x0, fs0, maxTimeSec);
if fs0 ~= fsTarget
    x0 = resample(x0, fsTarget, fs0);
    fs0 = fsTarget;
end
if ~isempty(snrTarget)
    x0 = addNoiseToSignal(x0, snrTarget);
end

S0 = melSpectrogram(x0, fs0, ...
    "WindowLength", frameLen, ...
    "OverlapLength", frameLen-hopLen, ...
    "NumBands", numMelBands);
spec0 = log10(S0 + eps);
[numBands, numFrames] = size(spec0);

X4d = zeros(numBands, numFrames, 1, numFiles, "single");

for i = 1:numFiles
    [x, fs] = audioread(ads.Files{i});
    x = mean(x,2);
    x = fixLength(x, fs, maxTimeSec);
    if fs ~= fsTarget
        x = resample(x, fsTarget, fs);
        fs = fsTarget;
    end
    if ~isempty(snrTarget)
        x = addNoiseToSignal(x, snrTarget);
    end

    S = melSpectrogram(x, fs, ...
        "WindowLength", frameLen, ...
        "OverlapLength", frameLen-hopLen, ...
        "NumBands", numMelBands);
    spec = log10(S + eps);   % bands x frames

    spec = imresize(spec, [numBands numFrames]);

    X4d(:,:,1,i) = single(spec);
end
end
