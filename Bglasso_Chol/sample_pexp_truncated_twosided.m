function r = sample_pexp_truncated_twosided(a,b,below,above)
% Sample truncated distribution 
%  \int_{x1}^{x2} x^(a-1) exp(bx) 1_{below<x<above} using Newton method and
% composite simposon's method(Eq (6.25) Introduction to Numerical Analysis
% Doron Levy)

u = (exp(log_pexp_pdf(a,b,0,above))-exp(log_pexp_pdf(a,b,0,below)))*rand(1)+ exp(log_pexp_pdf(a,b,0,below));


iter = 1;
delta = 1e-3;

q0 = above; %% Initialized at the upper boundary.
q1 = q0 - (exp(log_pexp_pdf(a,b,0,q0))-u)./(q0^(a-1)*exp(b*q0)); 
maxdiff = max(abs(q1-q0));
q0 = q1;

itermax = 100;
while (iter<itermax & maxdiff>delta)
iter  = iter+1;

q1 = q0 - (exp(log_pexp_pdf(a,b,0,q0))-u)./(q0^(a-1)*exp(b*q0)); 
maxdiff = max(abs(q1-q0));
q0 = q1;
%fprintf('iter=%d  maxdiff=%f \n',iter,maxdiff)
end

r = q1;