function [flag] = zero_test(d, alpha)
    low = prctile(d,alpha);
    high = prctile(d,100-alpha);
    if  (high < 0 || low > 0)
        flag = 1;
    else
        flag = 0;
    end
end