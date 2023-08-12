clc;
clear; 
close all;

%% Recursive LS
load('new.mat');
numParam = 1;
intial_parameter = -0.6061;
X = recursiveLS(numParam,'InitialParameters',-0.6);
N = length(y);
estimatedOut = zeros(1,N-1);
theta_est = zeros(1,N-1);
input = y(1);
for i = 1:N
    input(1) = 0;
    [theta, EstimatedOutput] = X(y(i),input(i));
    estimatedOut(i)= EstimatedOutput;
    theta_est(i) = theta;
    input(i+1) = y(i);
end

y = y(1:N);
e = estimatedOut - y;
plot(e);
N = length(e);
mu_0 = 0;
k_0 = 1;
alpha_0 = 10;
beta_0 = 1;
lambda_cp = 50;
H0 = 1/lambda_cp;
H1 = 1-H0;
e_bar = 0;

R                  = zeros(N+1,N+1);
R(1,1)             = 1;
post_pred_prob     = zeros(N+1,N+1);
mu     = zeros(N+1,N+1);
k      = zeros(N+1,N+1);
alpha  = zeros(N+1,N+1);
beta   = zeros(N+1,N+1);

mu(1,:)    = mu_0;
k(1,:)     = k_0;
alpha(1,:) = alpha_0;
beta(1,:)  = beta_0;
index = 1;

for t = 1:N
   % UPM Predictive
    for a = 1:t-index+1
        sigma            = sqrt((beta(a,t)*(k(a,t)+1))/ (alpha(a,t)*k(a,t))); 
        param1           = (e(t)-mu(a,t))/sigma;
        param2           = 2*alpha(a,t);
        post_pred_prob(a,t) = tpdf(param1,param2);
%        post_pred_prob(a,t) = gampdf(1/var(e(t+150:t+150+50)),alpha(a,t),(1/beta(a,t)));
    end
%%    Growth point probabiity
    for r = 1:t+1-index
       R(r+1,t+1) = post_pred_prob(r,t) * H1 * R(r,t);       
    end
    
    %% Change point probability
    change_point_prob = 0;
    for b = 1:t+1-index
       change_point_prob = change_point_prob + (post_pred_prob(1,b)*H0*R(b,t)); 
    end
    R(1,t+1) = change_point_prob;
    %% Normalization
    R(:,t+1) = R(:,t+1) / sum(R(:,t+1));
    %% Parameters Update         
   for r = 2:t+2-index
        e_bar = mean(e(t-r+2:t)); 
        var_e = sum(((e(index:t)-e_bar).^2));  
        mu(r,t+1) = (k(index,t)*mu(index,t) + (t+1-index)*e_bar)/(k(index,t) + (t+1-index));
        k(r,t+1) = k(index,t)+(t+1-index);
        alpha(r,t+1) = alpha(index,t)+((t+1-index)/2);
        beta(r,t+1) = beta(index,t) + var_e/2 + (k(index,t)*(t+1-index)*(e_bar-mu(index,t))^2)/(2*(k(index,t)+(t+1-index)));
   end   
   %% Detecting Change point and Updating starting point
   [val,max_R] = max(R(:,t));
   if(max_R<=2 && t>2)
        index = t;
        mu(1,t+1:N+1) =  mu(max_R,t);
        alpha(1,t+1:N+1) =  alpha(max_R,t);
        k(1,t+1:N+1) =  k(max_R,t);
        beta(1,t+1:N+1) =  beta(max_R,t);
   end
end
[value,RL] = max(R);