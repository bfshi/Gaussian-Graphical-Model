function [Specificity, Sensitivity, MCC] = eval_str(G_est, G)
[p,p2]=size(G);
epsilon = 1e-24;
TP = sum(sum(G_est.* G));
TN = sum(sum((1-G_est).*(1-G)));
FP = sum(sum(G_est.*(1-G)));
FN = sum(sum((1-G_est).*G));
Specificity = TN/(TN+FP+epsilon);
Sensitivity = TP/(TP+FN+epsilon);
MCC = (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+epsilon);
end