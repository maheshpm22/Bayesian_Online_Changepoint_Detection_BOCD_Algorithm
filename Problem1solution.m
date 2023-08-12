clc;
clear all;
close all;
set(0,'DefaultAxesFontWeight','bold','DefaultAxesFontSize', 14,'DefaultLineLineWidth', 2);
load("NMRlogWell.mat");

% Initial Conditions of Hyper Parameters
mu0 = 1.15;k0 = 0.01;alpha0 = 20;beta0 = 2;
lambdaCP = 250;

% Step1: Setting priors and initialization
T = numel(y);
R = zeros(T+1,T+1);R_plot = zeros(T+1,T+1);
R(1,1) = 1; % Assuming change point occurs before the data comes in
R_plot(1,1) = 1;
H = 1/lambdaCP;

mu = zeros(T+1,T+1);beta = zeros(T+1,T+1);alpha = zeros(T+1,T+1);
k = zeros(T+1,T+1);upm_predictive = zeros(T+1,T+1);

mu(:,1) = mu0;
alpha(:,1) = alpha0;
beta(:,1) = beta0;
k(:,1) = k0;
runLength = [];
str_idx = 1;

% Prediction loop(t=1:T)
for t=1:T
    % Step2: Observe new datum
    x = y(t);
    
    % Step3: Compute Predictive Proabilities of t-1
    for prev_r = 0:t-str_idx
        var_t = (beta(t,prev_r+1)*(k(t,prev_r+1)+1))/(alpha(t,prev_r+1)*k(t,prev_r+1));
        upm_predictive(t,prev_r+1) =  tpdf((x-mu(t,prev_r+1))/sqrt(var_t),2*alpha(t,prev_r+1))/sqrt(var_t);
    end
    
    % Step4: Compute Growth Probability of t
    R(t+1,2:t+2-str_idx) = upm_predictive(t,1:t+1-str_idx) .* (1-H) .* R(t,1:t+1-str_idx);
    
    % Step5: Compute Change Point Probability of t
    R(t+1,1) = sum(upm_predictive(t,1:t+1-str_idx) .* H .* R(t,1:t+1-str_idx));
    
    % Step6: Compute Evidence
    evidence = sum(R(t+1,:));
    
    % Step7: Compute RL Posterior
    R(t+1,:) = R(t+1,:)/evidence;
    
    
    % Step8: Checking for Change point
    [val,I] = sort(R(t,1:t),'descend');
    R_plot(t,I) = 1:t;
    runLength = [runLength I(1)];
    
    % Step8: Update parameters
    for r = 2:t+2-str_idx
        m = mean(y(t-r+2:t));
        mu(t+1,r) = (k(t,str_idx)*mu(t,str_idx) + (r-1)*m)/(k(t,str_idx) + (r-1));
        k(t+1,r) = k(t,str_idx)+(r-1);
        alpha(t+1,r) = alpha(t,str_idx)+((r-1)/2);
        beta(t+1,r) = beta(t,str_idx) + sum((y(str_idx:t)-m).^2)/2 + (k(t,str_idx)*(r-1)*(m-mu(t,str_idx))^2)/(2*(k(t,str_idx)+(r-1)));
    end
    
    if (I(1) == 3) && (t+1-str_idx)>7
        str_idx = t;
%         R(str_idx+1,2:end) = 0;
%         R(str_idx+1,1) = val(1);
%         mu(str_idx+1:T+1,:) = 0;
        mu(str_idx+1:T+1,1) =  mu(str_idx+1,I(1));
%         alpha(str_idx+1:T+1,:) = 0;
        alpha(str_idx+1:T+1,1) =  alpha(str_idx+1,I(1));
%         k(str_idx+1:T+1,:) = 0;
        k(str_idx+1:T+1,1) =  k(str_idx+1,I(1));
%         beta(str_idx+1:T+1,:) = 0;
        beta(str_idx+1:T+1,1) =  beta(str_idx+1,I(1));
    end
     
end

%% Results
for i=1:length(runLength)
    mu_plot(i) = mu(i,runLength(i));
end

figure;plot(y);hold on;plot(mu_plot);
legend('Data','Mean');
xlabel('Time');ylabel('Value')
title('Calculated Mean along with data');

figure;
subplot(211);
plot(y);xlabel('Time');ylabel('Value');
title('NMR Log Well Data');

subplot(212);hold on;
for i=1:100
    [a,b,~] = find(R_plot==i);
    colInt = (i-1)/100;
    plot(a,b,'.','Color',[0,0,0]+colInt);
end
plot(runLength,'r','linewidth',5);
colorbar('Direction','reverse','Ticks',[0,0.25,0.5,0.75,1],...
         'TickLabels',{'Prob = 1','Prob = 0.75','Prob = 0.5','Prob = 0.25','Prob = 0'});
colormap gray;
xlabel('Time');ylabel('Run Length');
title('RL Posterior Probability');