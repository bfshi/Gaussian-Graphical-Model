function [G] = graph_est(C,alpha)
[p1,p2,m] = size(C);
G = ones(p1);
for i = 1:p1
    for j = i+1:p1
        G(i,j) = zero_test(C(i,j,:),alpha);
        G(j,i) = G(i,j);
    end
end
end