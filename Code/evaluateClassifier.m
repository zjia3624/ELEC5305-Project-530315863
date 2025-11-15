function [yPred, acc, confMat] = evaluateClassifier(model, X, yTrue)
yPred  = predict(model, X);
acc    = mean(yPred == yTrue);
confMat = confusionmat(yTrue, yPred);
end
