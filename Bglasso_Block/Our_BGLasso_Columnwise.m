function [Sig_save,C_save,lambda_save] = Our_BGLasso_Columnwise(S,n,Sig,C,a_lambda,b_lambda,nu_0,sigma_0_square,burnin,nmc)
% Efficient Bayesian Graphical Lasso MCMC sampler using data-augmented
% block (column-wise) Gibbs sampler
%Input:
%     S = Y'*Y : sample covariance matrix * n
%     n: sample size
%     lambda:   |C|^(n/2) exp(-1/2 (SC) -  lambda/2 ||C||_1 );
%     Sig,C: initial Covariance and precision matrix C = inv(Sig);
%     burnin, nmc : number of MCMC burnins and saved samples
%     lambda ~ Ga(a_lambda,b_lambda)

%Output:
%     Sig_save: p by p by nmc matrices of saved posterior samples of covariance
%     C_save: p by p by nmc matrices of saved posterior samples of precision
%     lambda: 1 by nmc vector of saved posterior samples of lambda

%  Ref:  Wang 2012 Bayesain Graphical Lasso and Efficient Posterior
%  Computation

%  Written by Hao Wang & U of South Carolina

[p] = size(S,1); indmx = reshape([1:p^2],p,p); 
upperind = indmx(triu(indmx,1)>0); 

indmx_t = indmx';
lowerind = indmx_t(triu(indmx_t,1)>0); 

nu_0 = ones(length(C(upperind)), 1) * nu_0;
sigma_0_square = ones(length(C(upperind)), 1) * sigma_0_square;


C_save = zeros(p,p,nmc); Sig_save = C_save;
lambda_save = zeros(1,nmc);
 tau = zeros(p);

ind_noi_all = zeros(p-1,p);
for i = 1:p
       if i==1  
       ind_noi = [2:p]'; 
      elseif i==p
       ind_noi = [1:p-1]'; 
      else
       ind_noi = [1:i-1,i+1:p]';
       end
       
       ind_noi_all(:,i) = ind_noi;
end

    apost = a_lambda + p*(p+1)/2; 


for iter = 1: burnin+nmc    
            
       if(mod(iter,1000)==0)
        fprintf('iter = %d \n',iter);
       end

       
    % %%% Sample lambda 
    bpost = b_lambda + sum(abs(C(:)))/2;    
    lambda = gamrnd(apost,1/bpost,1);
    
%%% sample tau off-diagonal        
    temp = gamrnd((nu_0 + 1.) / 2., 2. ./ (nu_0 .* sigma_0_square + C(upperind) .* C(upperind)), [length(C(upperind)), 1]);
    tau_temp = ones(length(C(upperind)), 1) ./ (temp + 10^-6);
    %mean(tau_temp)

    tau(upperind) = tau_temp;
    tau(lowerind) = tau_temp;
    

%%% sample Sig and C = inv(Sig)        
    for i = 1:p


      ind_noi = ind_noi_all(:,i);
 
       
      tau_temp = tau(ind_noi,i);
       
      Sig11 = Sig(ind_noi,ind_noi); Sig12 = Sig(ind_noi,i);
      
      invC11 = Sig11 - Sig12*Sig12'/Sig(i,i);
      
      Ci = (S(i,i)+lambda)*invC11+diag(1./tau_temp);
  
        
        Ci_chol = chol(Ci);
        
        mu_i = -Ci\S(ind_noi,i);

        beta = mu_i+ Ci_chol\randn(p-1,1);
        
        
        C(ind_noi,i) = beta;
        C(i,ind_noi) = beta;
        gam = gamrnd(n/2+1,(S(i,i)+lambda)\2);
        
        C(i,i) = gam+beta'*invC11*beta;
    
        
        %% Below updating Covariance matrix according to one-column change of precision matrix
        invC11beta = invC11*beta;
        
        Sig(ind_noi,ind_noi) = invC11+invC11beta*invC11beta'/gam;
        Sig12 = -invC11beta/gam;
        Sig(ind_noi,i) = Sig12;
        Sig(i,ind_noi) = Sig12';
        Sig(i,i) = 1/gam;
        
    end


    

     


      
      
       if iter >burnin           
            Sig_save(:,:,iter-burnin) = Sig; 
            C_save(:,:,iter-burnin) = C;
            lambda_save(iter-burnin) = lambda;
       end



end


