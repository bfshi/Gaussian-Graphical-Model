function f = log_pexp_pdf(a,b,x1,x2)

% Log of the Composite simpson method for numerically integration: 
% log(\int_{x1}^{x2} x^(a-1) exp(bx))
% Reference: Eq (6.25) Introduction to Numerical Analysis Doron Levy

n = 100; % number of intervals;
h = (x2-x1)/n;
x = repmat(x1,1,n)+h*[1:n];

%fx = x.^(repmat(a,1,n)-1).*exp(repmat(b,1,n).*x);
%f = h/3.* (fx(:,1)+fx(:,end)+ 2*sum(fx(:,2:2:end-2),2)+4*sum(fx(:,1:2:end-1),2))



logfx = (repmat(a,1,n)-1).*log(x) + repmat(b,1,n).*x; 
logfx_max = max(logfx,[],2);
logfx2 = logfx-repmat(logfx_max,1,n);
fx = exp(logfx2);
f = h/3.* (fx(:,1)+fx(:,end)+ 2*sum(fx(:,2:2:end-2),2)+4*sum(fx(:,1:2:end-1),2));
f = logfx_max+log(f);