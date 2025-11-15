function [X, y] = extractFrameFeatures(ads, fsTarget, featType, ...
    numCoeffs, useDelta, snrTarget, frameLen, hopLen, maxTimeSec)
% Extract frame-based features (MFCC or log-mel) and average over time.
%  ads       : audioDatastore with Labels
%  featType  : "mfcc" or "logmel"
%  numCoeffs : base feature dimension (for mfcc or mel bands)
%  useDelta  : append delta & delta-delta if true
%  snrTarget : SNR in dB ([] = clean)
%  frameLen, hopLen, maxTimeSec : STFT params & fixed signal length

reset(ads);
numFiles = numel(ads.Files);
y = ads.Labels;

% First, use the first voice recording to determine the true feature dimensions.
[x0, fs0] = audioread(ads.Files{1});
x0 = mean(x0,2);

x0 = fixLengthLocal(x0, fs0, maxTimeSec);
if fs0 ~= fsTarget
    x0 = resample(x0, fsTarget, fs0);
    fs0 = fsTarget;
end
if ~isempty(snrTarget)
    x0 = addNoiseToSignal(x0, snrTarget);
end

featMat0 = computeFeatMat(x0, fs0, featType, numCoeffs, useDelta, frameLen, hopLen);
featDim  = size(featMat0, 2);

X = zeros(numFiles, featDim);
X(1,:) = mean(featMat0, 1);

% Then repeat the process on the remaining samples. 
for i = 2:numFiles
    [x, fs] = audioread(ads.Files{i});
    x = mean(x,2);
    x = fixLengthLocal(x, fs, maxTimeSec);

    if fs ~= fsTarget
        x = resample(x, fsTarget, fs);
        fs = fsTarget;
    end
    if ~isempty(snrTarget)
        x = addNoiseToSignal(x, snrTarget);
    end

    featMat = computeFeatMat(x, fs, featType, numCoeffs, useDelta, frameLen, hopLen);

    if size(featMat,2) ~= featDim
        error("Feature dimension mismatch: expected %d, got %d for file %s", ...
            featDim, size(featMat,2), ads.Files{i});
    end

    X(i,:) = mean(featMat, 1);
end

end

%  Auxiliary function: Calculates the feature matrix of a single speech segment.
function featMat = computeFeatMat(x, fs, featType, numCoeffs, useDelta, frameLen, hopLen)

switch featType
    case "mfcc"
        coeffs = mfcc(x, fs, ...
            "NumCoeffs", numCoeffs, ...
            "WindowLength", frameLen, ...
            "OverlapLength", frameLen-hopLen);
        % coeffs: frames x numCoeffs (在你这次出错的环境，很可能是 frames x (numCoeffs+1))
    case "logmel"
        S = melSpectrogram(x, fs, ...
            "WindowLength", frameLen, ...
            "OverlapLength", frameLen-hopLen, ...
            "NumBands", numCoeffs);
        coeffs = log10(S + eps).';   % frames x numBands
    otherwise
        error("Unsupported featType: %s", featType);
end

if useDelta
    d1 = diff([coeffs(1,:); coeffs],1,1);
    d2 = diff([d1(1,:); d1],1,1);
    featMat = [coeffs d1 d2];    % frames x (D * 3)
else
    featMat = coeffs;
end

end

%  Auxiliary functions: uniform length
function xOut = fixLengthLocal(x, fs, maxTimeSec)
Ntarget = round(maxTimeSec * fs);
N = numel(x);
if N > Ntarget
    xOut = x(1:Ntarget);
elseif N < Ntarget
    xOut = [x; zeros(Ntarget-N,1)];
else
    xOut = x;
end
end
