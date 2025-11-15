function y = addNoiseToSignal(x, targetSNRdB)
signalPower = mean(x.^2);
snrLinear   = 10^(targetSNRdB/10);
noisePower  = signalPower / snrLinear;
noise = sqrt(noisePower) * randn(size(x));
y = x + noise;
end
