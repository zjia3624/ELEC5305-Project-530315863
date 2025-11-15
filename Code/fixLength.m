function xOut = fixLength(x, fs, maxTimeSec)
%FIXLENGTH Trim or zero-pad signal to a fixed duration.
%   x          : input signal (column vector)
%   fs         : sample rate (Hz)
%   maxTimeSec : target duration (seconds)

Ntarget = round(maxTimeSec * fs);
N = numel(x);

if N > Ntarget
    % Cut off
    xOut = x(1:Ntarget);
elseif N < Ntarget
    % Padding with zeros at the end
    xOut = [x; zeros(Ntarget-N, 1)];
else

    xOut = x;
end
end
