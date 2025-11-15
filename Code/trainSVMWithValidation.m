function [bestModel, bestCfg] = trainSVMWithValidation(XTrain, yTrain, params)
kernelScaleList   = params.kernelScaleList;
boxConstraintList = params.boxConstraintList;

bestAcc  = -inf;
bestModel = [];
bestCfg   = struct("kernelScale", [], "boxConstraint", []);

for ks = kernelScaleList
    for C = boxConstraintList
        t = templateSVM("KernelFunction","rbf", ...
                        "KernelScale",ks, ...
                        "BoxConstraint",C, ...
                        "Standardize",true);

        mdlCV = fitcecoc(XTrain, yTrain, ...
            "Learners",t, "Coding","onevsall", "KFold",5);

        cvLoss = kfoldLoss(mdlCV);
        cvAcc  = 1 - cvLoss;
        fprintf("  ks=%.2f, C=%.2f -> CV acc=%.3f\n", ks, C, cvAcc);

        if cvAcc > bestAcc
            bestAcc = cvAcc;
            bestModel = fitcecoc(XTrain, yTrain, ...
                "Learners",t, "Coding","onevsall");
            bestCfg.kernelScale   = ks;
            bestCfg.boxConstraint = C;
        end
    end
end
fprintf("Best CV acc = %.3f\n", bestAcc);
end
