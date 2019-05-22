function [Sig_save,C_save] = BayesGLassoGDP(S,n,Sig,C,burnin,nmc)
% Efficient Bayesian Adaptive Graphical Lasso (or generalized double Pareto)  MCMC sampler using data-augmented
% block(column) Gibbs sampler
%
%     y \sim N(0, C^{-1}), Y = (y_1,...,y_n), S = Y*Y';     p(Y) \propto |C|^(n/2) exp(-1/2 (SC) )
%     C_jj  \sim Exp(lamba_ii/2);                                p(C_jj) \prop \lambda_ii exp(- \lambda_ii/2 C_ii)
%     C_{ij} \sim DE(lambda_{ij} ),                      p(C_{ij} \propto exp(- lambda_{ij} |C_{ij}| )
%     lambda_{ij} \sim Ga(s,t).                   

%  This adaptive glasso prior is equivalent to generalized double Pareto: 

%     y \sim N(0, C^{-1}), Y = (y_1,...,y_n), S = Y*Y';  |C|^(n/2) exp(-1/2 (SC) )
%     C_ii   \sim  Exp(lamba_ii/2 );    
%     C_{ij} \sim  GDP( xi = t/s, alpha = s) where x \sim GDP(xi,alpha) has density
%     f(x) = 1/(2 xi)(1+ |x|/(alpha xi))^{-(1+alpha)}   


%Input:
%     S = Y'*Y : sample covariance matrix * n
%     n: sample size
%     Sig,C: initial Covariance and precision matrix C = inv(Sig);
%     burnin, nmc : number of MCMC burnins and saved samples

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


C_save = zeros(p,p,nmc); Sig_save = C_save;


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


%% Hyperparameter of \lambda_{ij} ~ Ga(s,t)

    s = 1e-2;  
    t =1e-6;
    
    lambda_ii = 1;
    
for iter = 1: burnin+nmc    
     

     
       if(mod(iter,1000)==0)
        fprintf('iter = %d \n',iter);
       end
       
   Cadjust = max(abs(C(upperind)),10^-12);        

       


    %%% sample lambda off-diagonal
    s_post = 1+s;  t_post = Cadjust+t;
    lambda = gamrnd(s_post,1./t_post);                
     
    
%%% sample tau off-diagonal        
    lambda_prime = lambda.^2;  mu_prime = min(lambda./Cadjust,1e12);
    tau_temp =  max(1e-12,1./rand_ig(mu_prime,lambda_prime));
    tau(upperind) = tau_temp;
    tau(lowerind) = tau_temp;
    
    
  
    
    
    
    
    

%%% sample Sig and C = inv(Sig)        
    for i = 1:p


      ind_noi = ind_noi_all(:,i);
 
       
      tau_temp = tau(ind_noi,i);
       
      Sig11 = Sig(ind_noi,ind_noi); Sig12 = Sig(ind_noi,i);
      
      invC11 = Sig11 - Sig12*Sig12'/Sig(i,i);
      
      Ci = (S(i,i)+lambda_ii)*invC11+diag(1./tau_temp);
        
        
      Ci = (Ci+Ci')./2; Ci_chol = chol(Ci);
        
        mu_i = -Ci\S(ind_noi,i);

        beta = mu_i+ Ci_chol\randn(p-1,1);
        
        
        C(ind_noi,i) = beta;
        C(i,ind_noi) = beta;
        gam = gamrnd(n/2+1,(S(i,i)+lambda_ii)\2);
        
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
       end



end


