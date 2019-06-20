addpath ./Bglasso_Block/
addpath ./Bglasso_Chol/
addpath ./Fglasso/

%% This Demo illustrates the simulation study of Section 5 in Wang(2012)
%% It includes six different cases for underlying covariance matrics and 
%%  five graphical models: 
%%  glasso, adaptive glasso, scad, Bayes glass and Bayes adaptive glasso.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%  Six Cases     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p = 30; n = 50;

%%% AR(1) case
SigTrue = toeplitz(0.7.^[0:p-1]);

%%%% AR(2) case 
% CTrue = toeplitz([1,0.5,0.25,zeros(1,p-3)]);
% SigTrue = inv(CTrue);
% 
% 
% %%%% Block case
% SigTrue = eye(p);
% SigTrue(1:p/2,1:p/2) = 0.5*ones(p/2)+(1-0.5)*eye(p/2);
% SigTrue(p/2+1:end,p/2+1:end) = 0.5*ones(p/2)+(1-0.5)*eye(p/2);
% 
% 
% %%% Star case
% CTrue = eye(p); CTrue(1,2:end) = 0.1; CTrue(2:end,1) = 0.1;
% SigTrue = inv(CTrue);
% 
% 
% %%% Circle case
% SigTrue = inv(toeplitz([2,1,zeros(1,p-3),0.9]));
% CTrue = inv(SigTrue); 
% 
% %%% Full case
% CTrue = ones(p)+eye(p);
% SigTrue = inv(CTrue);
% 
CTrue = inv(SigTrue);
threshold = 1e-5;
G = (CTrue>threshold)+(CTrue<-threshold);
p = size(SigTrue,1);
n = 50;

Y = rMNorm(zeros(p,1),SigTrue,n);
S = Y*Y';

alpha = 2.5;

indmx = reshape([1:p^2],p,p); 
upperind = indmx(triu(indmx,1)>0); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%  Five methods     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

burnin  = 1000; nmc = 2000;

%% (0) Our Bayesian Graphical Lasso
a_lambda = 1; b_lambda = 0.1; % Prior hyperparameters
Sig = S/n; C = inv(Sig); % Initial values 
C_square = C.*C;
E_sigma_square = mean(C_square(upperind));
Var_sigma_square = var(C_square(upperind));
%%% moment estimation
nu_0 = 2. * E_sigma_square^2 / Var_sigma_square + 4.;
sigma_0_square = (nu_0 - 2) * E_sigma_square / nu_0 / 50.;  % 50 is a hyperparameter

 %nu_0 = 1000.; sigma_0_square = 0.001;
 nu_0
 sigma_0_square

[Sig_save,C_save,lambda_save] = Our_BGLasso_Columnwise(S,n,Sig,C,a_lambda,b_lambda,nu_0,sigma_0_square,burnin,nmc);
G_est = graph_est(C_save, alpha);


%%%  Run the slower Cholesky-based Gibbs sampler to verify the correctness of the block
%%%  and Cholesky-based Gibbs samplers by inspecting the agreement of the posterior summaries of
%%%  parameters from these two different samplers ...
%  [T_save,d_save,C_save_chol,Sig_save_chol,lambda_save_chol] = Bglasso_mcmc_chol_lambdaunknown(S,n,a_lambda,b_lambda,C,burnin,nmc);


%Stein's loss and its Bayes estimates
Sig_L1_bglasso = inv(mean(C_save,3));
L1_our_bglasso    = trace(Sig_L1_bglasso/SigTrue) - log(det(Sig_L1_bglasso/SigTrue)) - p;
[our_Specificity, our_Sensitivity, our_MCC] = eval_str(G_est, G);

%%%%% GHS
[C_save,lambda_sq_save,tau_sq_save] = GHS(S,n,burnin,nmc);
Sig_L1_bglasso = inv(mean(C_save,3));
G_est = graph_est(C_save, alpha);
L1_GHS_bglasso    = trace(Sig_L1_bglasso/SigTrue) - log(det(Sig_L1_bglasso/SigTrue)) - p;
[GHS_Specificity, GHS_Sensitivity, GHS_MCC] = eval_str(G_est, G);

%%%%% (1) Bayesian Graphical Lasso (Friedman et al 2009 Biostatistics)
a_lambda = 1; b_lambda = 0.1; % Prior hyperparameters
Sig = S/n; C = inv(Sig); % Initial values 
[Sig_save,C_save,lambda_save] = BayesGLasso_Columnwise(S,n,Sig,C,a_lambda,b_lambda,burnin,nmc);
G_est = graph_est(C_save, alpha);

%%  Stein's loss and its Bayes estimates
Sig_L1_bglasso = inv(mean(C_save,3));
L1_bglasso    = trace(Sig_L1_bglasso/SigTrue) - log(det(Sig_L1_bglasso/SigTrue)) - p;
[bg_Specificity, bg_Sensitivity, bg_MCC] = eval_str(G_est, G);
% 
% 
%%%%% (2) Bayesian Adaptive Graphical Lasso (i.e., generalized Double
%%%%% Pareto with fixed hyperparameter)
Sig = S/n; C = inv(Sig); % Initial values 
[Sig_save,C_save] = BayesGLassoGDP(S,n,Sig,C,burnin,nmc);
Sig_L1_bgdp = inv(mean(C_save,3));
L1_bgdp = trace(Sig_L1_bgdp/SigTrue) - log(det(Sig_L1_bgdp/SigTrue)) - p;
G_est = graph_est(C_save, alpha);
[bgdp_Specificity, bgdp_Sensitivity, bgdp_MCC] = eval_str(G_est, G);


%%%% (3) Frequentisit Glasso+CV (Friedman et al 2009 Biostatistics)
  rho_mean = mean(lambda_save)/n*2;
  d = rho_mean/5;
  rhopath = [rho_mean/5:d:rho_mean*3]; fold  = 10;
  [W_lasso,M_lasso,rhomax_lasso] = glasso_cv(Y,rhopath,fold);


  % Change grids if rhomax is on the boundary of the previous "rhopath".
      if rhomax_lasso == rhopath(end)    
          while rhomax_lasso == rhopath(end)
          rhopath = [rhopath(end):d:rhopath(end)+3*rho_mean];
          [W_lasso,M_lasso,rhomax_lasso] = glasso_cv(Y,rhopath,fold);
          end
          
      elseif rhomax_lasso == rhopath(1)
         
          while rhomax_lasso == rhopath(1)          
          d1 = rhopath(1)/8;
          rhopath = [rhopath(1)/2:d1:rhopath(1)];
          [W_lasso,M_lasso,rhomax_lasso] = glasso_cv(Y,rhopath,fold);          
          end
      end
      

  
L1_glasso = trace(W_lasso/SigTrue) - log(det(W_lasso/SigTrue)) - p;
G_est = inv(W_lasso);
G_est = (G_est>0.001)+(G_est<-0.001);
[g_Specificity, g_Sensitivity, g_MCC] = eval_str(G_est, G);


  
%%%% (4) Frequentisit AdapGlasso+CV  (Fan Feng Wu 2009 AoAs)
rho_mean = mean(lambda_save)/n*2;
  d = rho_mean/5;
  rhopath = [rho_mean/5:d:rho_mean*3]; fold  = 10;
  [W_adapt,M_adapt,M0,rhomax_adapt] = glasso_adaptive_cv(Y,rhopath,fold);
  
          if rhomax_adapt == rhopath(end)
          while rhomax_adapt == rhopath(end)
          rhopath = [rhopath(end):d:rhopath(end)+3*rho_mean];
          [W_adapt,M_adapt,M0,rhomax_adapt] = glasso_adaptive_cv(Y,rhopath,fold);
          end
          
      elseif rhomax_adapt == rhopath(1)
         
          while rhomax_adapt == rhopath(1)          
          d1 = rhopath(1)/8;
          rhopath = [rhopath(1)/2:d1:rhopath(1)];
          [W_adapt,M_adapt,M0,rhomax_adapt] = glasso_adaptive_cv(Y,rhopath,fold);          
          end
          end
        
          
L1_adapt = trace(W_adapt/SigTrue) - log(det(W_adapt/SigTrue)) - p;
G_est = inv(W_adapt);
G_est = (G_est>0.001)+(G_est<-0.001);
[adpt_Specificity, adpt_Sensitivity, adpt_MCC] = eval_str(G_est, G);
  
%%%% (5) Graphical SCAD+CV (Fan Feng Wu 2009 AoAs)   
  rhopath = [9*rho_mean:d:rho_mean*12]; 
  [W_scad,M_scad,M0,rhomax_scad] = glasso_SCAD_cv(Y,rhopath,fold); 
  
        if rhomax_scad == rhopath(end)
          while rhomax_scad == rhopath(end)
          rhopath = [rhopath(end):d:rhopath(end)+3*rho_mean];
          [W_scad,M_scad,M0,rhomax_scad] = glasso_SCAD_cv(Y,rhopath,fold);
          end          
      elseif rhomax_scad == rhopath(1)         
          while rhomax_scad == rhopath(1)          
          d1 = rhopath(1)/8;
          rhopath = [rhopath(1)/2:d1:rhopath(1)];
          [W_scad,M_scad,M0,rhomax_scad] = glasso_SCAD_cv(Y,rhopath,fold);          
          end
        end
      
              
L1_scad = trace(W_scad/SigTrue) - log(det(W_scad/SigTrue)) - p;
G_est = inv(W_scad);
G_est = (G_est>0.001)+(G_est<-0.001);
[scad_Specificity, scad_Sensitivity, scad_MCC] = eval_str(G_est, G);
 
l1 = [L1_our_bglasso,L1_GHS_bglasso,L1_bglasso,L1_bgdp,L1_glasso,L1_adapt,L1_scad];
Sensitivity = [our_Sensitivity, GHS_Sensitivity,bg_Sensitivity, bgdp_Sensitivity, g_Sensitivity, adpt_Sensitivity, scad_Sensitivity];
Specificity = [our_Specificity, GHS_Specificity,bg_Specificity, bgdp_Specificity, g_Specificity, adpt_Specificity, scad_Specificity];
MCC = [our_MCC,GHS_MCC, bg_MCC, bgdp_MCC, g_MCC, adpt_MCC, scad_MCC];
