% Group 1

clear all; close all; clc;
addpath('mcmcstat-master');  

% True parameter values
beta_true  = 0.5;
gamma_true = 0.2;
phi_true   = 0.2;
kappa      = 0.0;  
theta_true = [beta_true, gamma_true, phi_true];
param_names = {'\beta', '\gamma', '\phi'};

% Population and initial conditions
N  = 100;
S0 = N - 10;  
I0 = 10;
R0 = 0;
y0 = [S0; I0; R0];

% Time grid
tspan = linspace(0, 50, 101);

% We solve the ODE with true parameters
[t_sol, y_true] = ode45(@sirs_ode, tspan, y0, [], theta_true, N, kappa);

% We add the gaussian noise
sigma_noise = 2.0;
rng(42);
y_noisy = y_true + sigma_noise * randn(size(y_true));

data.tspan = tspan;
data.ydata = y_noisy;
data.y0    = y0;
data.N     = N;
data.kappa = kappa;

% For the synthetic data
figure('Name','Synthetic Data','Position',[100 100 900 500]);
colors = {'b','r','g'};
names  = {'S','I','R'};
for i = 1:3
    plot(tspan, y_true(:,i), [colors{i},'-'], 'LineWidth', 2); hold on;
    plot(tspan, y_noisy(:,i), [colors{i},'o'], 'MarkerSize', 3);
end
xlabel('Time (days)'); ylabel('Number of individuals');
title('SIRS Model: True Solution and Noisy Synthetic Data');
legend('S (true)','S (noisy)','I (true)','I (noisy)','R (true)','R (noisy)','Location','best');
grid on;
saveas(gcf, 'fig01_synthetic_data.png');

% For the individual compartments 
figure('Name','Compartments','Position',[100 100 1200 400]);
for i = 1:3
    subplot(1,3,i);
    plot(tspan, y_true(:,i), [colors{i},'-'], 'LineWidth', 2); hold on;
    plot(tspan, y_noisy(:,i), [colors{i},'o'], 'MarkerSize', 3);
    xlabel('Time (days)'); ylabel('Count');
    title([names{i}, ' Compartment']);
    legend('True','Noisy data','Location','best'); grid on;
end

fprintf(' True params: beta=%.2f, gamma=%.2f, phi=%.2f\n\n', theta_true);

%  We do a sensitivity analysis for Kappa to check if a small change in it
%  causes a change in the model because it was specified to be zero

kappa_vals = [0, 0.001, 0.005, 0.01, 0.02, 0.05];

% We check the dynamics for different kappa
figure('Name','Kappa Sensitivity','Position',[100 100 1200 700]);
for idx = 1:length(kappa_vals)
    kv = kappa_vals(idx);
    [~, y_k] = ode45(@sirs_ode, tspan, y0, [], theta_true, N, kv);
    subplot(2,3,idx);
    plot(tspan, y_k(:,1), 'b-', tspan, y_k(:,2), 'r-', tspan, y_k(:,3), 'g-', 'LineWidth', 2);
    xlabel('Time'); ylabel('Count');
    title(sprintf('\\kappa = %.3f', kv));
    legend('S','I','R','Location','best'); grid on;
end
sgtitle('Sensitivity of SIRS Dynamics to \kappa');
saveas(gcf, 'fig05_kappa_sensitivity.png');

% We check SS anf the kappa their profile how they are
kappa_range = linspace(0, 0.1, 51);
ss_kappa = zeros(size(kappa_range));
for idx = 1:length(kappa_range)
    [~, yk] = ode45(@sirs_ode, tspan, y0, [], theta_true, N, kappa_range(idx));
    ss_kappa(idx) = sum(sum((y_noisy - yk).^2));
end

figure('Name','Kappa Profile','Position',[100 100 700 450]);
plot(kappa_range, ss_kappa, 'b-o', 'MarkerSize', 4, 'LineWidth', 2);
hold on;
xline(0, 'r--', 'LineWidth', 1.5);
xlabel('\kappa'); ylabel('Sum of Squares');
title('Profile: SS vs \kappa (other params at true values)');
legend('SS(\kappa)', 'True \kappa = 0', 'Location', 'best'); grid on;
saveas(gcf, 'fig06_kappa_profile.png');

fprintf('  Minimum SS at kappa=%.4f (SS=%.2f)\n', kappa_range(ss_kappa==min(ss_kappa)), min(ss_kappa));
fprintf('  SS at kappa=0: %.2f\n\n', ss_kappa(1));

% Least Squares Estimation

theta0 = [0.4, 0.15, 0.15]; 
options_fmin = optimset('MaxIter', 10000, 'TolFun', 1e-10, 'TolX', 1e-10, 'Display', 'off');

[theta_opt, ss_opt] = fminsearch(@(th) sirs_cost(th, data), theta0, options_fmin);
[~, y_fit] = ode45(@sirs_ode, tspan, y0, [], theta_opt, N, kappa);

fprintf('  LSQ estimates: beta=%.4f, gamma=%.4f, phi=%.4f\n', theta_opt);
fprintf('  Optimal SS = %.2f\n\n', ss_opt);

% LSQ Fit 
figure('Name','LSQ Fit','Position',[100 100 900 500]);
for i = 1:3
    plot(tspan, y_noisy(:,i), [colors{i},'o'], 'MarkerSize', 3); hold on;
    plot(tspan, y_fit(:,i), [colors{i},'-'], 'LineWidth', 2.5);
end
xlabel('Time (days)'); ylabel('Number of individuals');
title('Least Squares Fit to Noisy Data');
legend('S (data)','S (fit)','I (data)','I (fit)','R (data)','R (fit)','Location','best');
grid on;
saveas(gcf, 'fig03_lsq_fit.png');

% Residuals 
residuals = y_noisy - y_fit;
figure('Name',' Residuals','Position',[100 100 1200 700]);
for i = 1:3
    subplot(2,3,i);
    plot(tspan, residuals(:,i), [colors{i},'o'], 'MarkerSize', 3);
    hold on; yline(0,'k--','LineWidth',1);
    xlabel('Time'); ylabel('Residual');
    title([names{i}, ' Residuals vs Time']); grid on;
    
    subplot(2,3,i+3);
    histogram(residuals(:,i), 20, 'Normalization','pdf', 'FaceColor', colors{i});
    xlabel('Residual'); ylabel('Density');
    title([names{i}, ' Residual Distribution']); grid on;
end
sgtitle('Residual Analysis');

% We now check the Jacobian-Based Uncertainty Quantification
n_par = 3;
n_t   = length(tspan);
n_obs = n_t * 3;

J = compute_jacobian(theta_opt, data);

sigma2_est = ss_opt / (n_obs - n_par);
JtJ = J' * J;
C   = sigma2_est * inv(JtJ);
theta_std = sqrt(diag(C));
t_values  = theta_opt(:) ./ theta_std;

% Correlation matrix
D = diag(1 ./ theta_std);
corr_mat = D * C * D;

% R-squared
y_mean   = mean(y_noisy(:));
ss_total = sum((y_noisy(:) - y_mean).^2);
R2 = 1 - ss_opt / ss_total;

fprintf('  sigma^2 estimate = %.4f (true = %.1f)\n', sigma2_est, sigma_noise^2);
fprintf('  R^2 = %.6f\n', R2);
fprintf('  %-8s %-10s %-10s %-10s\n', 'Param', 'Estimate', 'Std', 't-value');
for i = 1:n_par
    fprintf('  %-8s %-10.4f %-10.4f %-10.2f\n', param_names{i}, theta_opt(i), theta_std(i), t_values(i));
end
fprintf('  Correlation matrix:\n');
disp(corr_mat);


% Correlation Heatmap for the correlation matrix
figure('Name','Correlation','Position',[100 100 450 400]);
imagesc(corr_mat); colorbar; caxis([-1 1]); colormap(flipud(jet));
set(gca, 'XTick', 1:3, 'XTickLabel', {'\beta','\gamma','\phi'});
set(gca, 'YTick', 1:3, 'YTickLabel', {'\beta','\gamma','\phi'});
for i = 1:3
    for j = 1:3
        text(j, i, sprintf('%.3f', corr_mat(i,j)), 'HorizontalAlignment','center','FontSize',12);
    end
end
title('Parameter Correlation Matrix');

%  MCMC with mcmcstat using the DRAM
mcmc_data.xdata = tspan(:);
mcmc_data.ydata = y_noisy;
mcmc_data.y0    = y0;
mcmc_data.N     = N;
mcmc_data.kappa = kappa;

model.ssfun    = @sirs_cost_mcmc;
model.sigma2   = sigma2_est;
model.N        = n_obs;

params = {
    {'beta',  theta_opt(1), 0, 5}
    {'gamma', theta_opt(2), 0, 5}
    {'phi',   theta_opt(3), 0, 5}
};

options.nsimu       = 10000;
options.method      = 'dram';
options.adaptint    = 100;
options.qcov        = C;
options.updatesigma = 1;
options.verbosity   = 1;

[results, chain, s2chain, sschain] = mcmcrun(model, mcmc_data, params, options);

% Burn-in
burnin = round(0.25 * size(chain,1));
chain_burned  = chain(burnin:end, :);
s2_burned     = s2chain(burnin:end, :);

for i = 1:n_par
    ch = chain_burned(:,i);
    ci = prctile(ch, [2.5, 97.5]);
end

% Trace Plots 
figure('Name','Trace','Position',[100 100 1000 800]);
for i = 1:3
    subplot(4,1,i);
    plot(chain(:,i), 'Color', [0.3 0.5 0.7]); hold on;
    yline(theta_true(i), 'r--', 'LineWidth', 1.5);
    xline(burnin, 'Color', [1 0.6 0], 'LineStyle', ':', 'LineWidth', 1.5);
    ylabel(param_names{i}); title(['Trace: ', param_names{i}]);
end
subplot(4,1,4);
plot(s2chain, 'Color', [0.5 0.2 0.7]); hold on;
yline(sigma_noise^2, 'r--', 'LineWidth', 1.5);
xline(burnin, 'Color', [1 0.6 0], 'LineStyle', ':', 'LineWidth', 1.5);
ylabel('\sigma^2'); xlabel('Iteration'); title('Trace: \sigma^2');
sgtitle('MCMC Trace Plots');


%  MCMC with mcmcstat using the AM for additional experiment
mcmc_data.xdata = tspan(:);
mcmc_data.ydata = y_noisy;
mcmc_data.y0    = y0;
mcmc_data.N     = N;
mcmc_data.kappa = kappa;

model.ssfun    = @sirs_cost_mcmc;
model.sigma2   = sigma2_est;
model.N        = n_obs;

params = {
    {'beta',  theta_opt(1), 0, 5}
    {'gamma', theta_opt(2), 0, 5}
    {'phi',   theta_opt(3), 0, 5}
};

options.nsimu       = 10000;
options.method      = 'am';
options.adaptint    = 100;
options.qcov        = C;
options.updatesigma = 1;
options.verbosity   = 1;

[results, chain, s2chain, sschain] = mcmcrun(model, mcmc_data, params, options);

% Burn-in
burnin = round(0.25 * size(chain,1));
chain_burned  = chain(burnin:end, :);
s2_burned     = s2chain(burnin:end, :);

for i = 1:n_par
    ch = chain_burned(:,i);
    ci = prctile(ch, [2.5, 97.5]);
end

% Trace Plots 
figure('Name','Trace','Position',[100 100 1000 800]);
for i = 1:3
    subplot(4,1,i);
    plot(chain(:,i), 'Color', [0.3 0.5 0.7]); hold on;
    yline(theta_true(i), 'r--', 'LineWidth', 1.5);
    xline(burnin, 'Color', [1 0.6 0], 'LineStyle', ':', 'LineWidth', 1.5);
    ylabel(param_names{i}); title(['Trace: ', param_names{i}]);
end
subplot(4,1,4);
plot(s2chain, 'Color', [0.5 0.2 0.7]); hold on;
yline(sigma_noise^2, 'r--', 'LineWidth', 1.5);
xline(burnin, 'Color', [1 0.6 0], 'LineStyle', ':', 'LineWidth', 1.5);
ylabel('\sigma^2'); xlabel('Iteration'); title('Trace: \sigma^2');
sgtitle('MCMC Trace Plots using Adaptive Metropolis');



%  MCMC with mcmcstat using the MH for additional experiment
mcmc_data.xdata = tspan(:);
mcmc_data.ydata = y_noisy;
mcmc_data.y0    = y0;
mcmc_data.N     = N;
mcmc_data.kappa = kappa;

model.ssfun    = @sirs_cost_mcmc;
model.sigma2   = sigma2_est;
model.N        = n_obs;

params = {
    {'beta',  theta_opt(1), 0, 5}
    {'gamma', theta_opt(2), 0, 5}
    {'phi',   theta_opt(3), 0, 5}
};

options.nsimu       = 10000;
options.method      = 'mh';
options.adaptint    = 100;
options.qcov        = C;
options.updatesigma = 1;
options.verbosity   = 1;

[results, chain, s2chain, sschain] = mcmcrun(model, mcmc_data, params, options);

% Burn-in
burnin = round(0.25 * size(chain,1));
chain_burned  = chain(burnin:end, :);
s2_burned     = s2chain(burnin:end, :);

for i = 1:n_par
    ch = chain_burned(:,i);
    ci = prctile(ch, [2.5, 97.5]);
end

% Trace Plots 
figure('Name','Trace','Position',[100 100 1000 800]);
for i = 1:3
    subplot(4,1,i);
    plot(chain(:,i), 'Color', [0.3 0.5 0.7]); hold on;
    yline(theta_true(i), 'r--', 'LineWidth', 1.5);
    xline(burnin, 'Color', [1 0.6 0], 'LineStyle', ':', 'LineWidth', 1.5);
    ylabel(param_names{i}); title(['Trace: ', param_names{i}]);
end
subplot(4,1,4);
plot(s2chain, 'Color', [0.5 0.2 0.7]); hold on;
yline(sigma_noise^2, 'r--', 'LineWidth', 1.5);
xline(burnin, 'Color', [1 0.6 0], 'LineStyle', ':', 'LineWidth', 1.5);
ylabel('\sigma^2'); xlabel('Iteration'); title('Trace: \sigma^2');
sgtitle('MCMC Trace Plots using Metropolis Hastings');



% Posterior Histograms 
figure('Name','Posteriors','Position',[100 100 1400 400]);
for i = 1:3
    subplot(1,4,i);
    histogram(chain_burned(:,i), 60, 'Normalization','pdf', 'FaceColor', [0.3 0.5 0.7]);
    hold on;
    xline(theta_true(i), 'r--', 'LineWidth', 2);
    xline(theta_opt(i), 'g--', 'LineWidth', 2);
    xlabel(param_names{i}); ylabel('Density');
    title(['Posterior: ', param_names{i}]);
    legend('Posterior','True','LSQ','Location','best');
end
subplot(1,4,4);
histogram(s2_burned, 60, 'Normalization','pdf', 'FaceColor', [0.5 0.2 0.7]);
hold on; xline(sigma_noise^2, 'r--', 'LineWidth', 2);
xlabel('\sigma^2'); ylabel('Density');
title('Posterior: \sigma^2');

% Pairwise Plot 
figure('Name','Pairwise Plot','Position',[100 100 800 800]);
thin = max(1, floor(size(chain_burned,1)/3000));
ct = chain_burned(1:thin:end, :);
for i = 1:3
    for j = 1:3
        subplot(3,3,(i-1)*3+j);
        if i == j
            histogram(ct(:,i), 50, 'Normalization','pdf', 'FaceColor',[0.3 0.5 0.7]);
            hold on; xline(theta_true(i),'r--','LineWidth',1.5);
        elseif i > j
            plot(ct(:,j), ct(:,i), '.', 'Color', [0.3 0.5 0.7], 'MarkerSize', 1);
            hold on; plot(theta_true(j), theta_true(i), 'r*', 'MarkerSize', 15);
        else
            axis off;
        end
        if i==3, xlabel(param_names{j}); end
        if j==1, ylabel(param_names{i}); end
    end
end
sgtitle('Pairwise Posterior Scatter Plots');


% Autocorrelation 
figure('Name',' ACF','Position',[100 100 1200 350]);
max_lag = 80;
for i = 1:3
    subplot(1,3,i);
    ch = chain_burned(:,i) - mean(chain_burned(:,i));
    acf_full = xcorr(ch, max_lag, 'coeff');
    acf_vals = acf_full(max_lag+1:end);  % positive lags only
    bar(0:max_lag, acf_vals, 'FaceColor', [0.3 0.5 0.7]);
    hold on;
    yline(1.96/sqrt(length(chain_burned(:,i))), 'r--');
    yline(-1.96/sqrt(length(chain_burned(:,i))), 'r--');
    xlabel('Lag'); ylabel('ACF');
    title(['ACF: ', param_names{i}]);
    ylim([-0.15, 1.05]);
end
sgtitle('MCMC Autocorrelation Functions');


% MCMC Predictive Distribution

n_pred = 500;
thin_p = max(1, floor(size(chain_burned,1)/n_pred));
idx_p  = 1:thin_p:size(chain_burned,1);
if length(idx_p) > n_pred, idx_p = idx_p(1:n_pred); end

y_samples = zeros(length(idx_p), n_t, 3);
for ii = 1:length(idx_p)
    th = chain_burned(idx_p(ii), :);
    [~, ys] = ode45(@sirs_ode, tspan, y0, [], th, N, kappa);
    y_samples(ii,:,:) = ys;
end

%  MCMC Predictive Bands 
figure('Name',' Predictive','Position',[100 100 1400 400]);
for i = 1:3
    subplot(1,3,i);
    q025 = prctile(squeeze(y_samples(:,:,i)), 2.5);
    q975 = prctile(squeeze(y_samples(:,:,i)), 97.5);
    q25  = prctile(squeeze(y_samples(:,:,i)), 25);
    q75  = prctile(squeeze(y_samples(:,:,i)), 75);
    med  = median(squeeze(y_samples(:,:,i)));
    fill([tspan, fliplr(tspan)], [q025, fliplr(q975)], colors{i}, 'FaceAlpha',0.15,'EdgeColor','none'); hold on;
    fill([tspan, fliplr(tspan)], [q25, fliplr(q75)], colors{i}, 'FaceAlpha',0.35,'EdgeColor','none');
    plot(tspan, med, [colors{i},'-'], 'LineWidth', 2.5);
    plot(tspan, y_noisy(:,i), 'ko', 'MarkerSize', 2.5);
    plot(tspan, y_true(:,i), 'k--', 'LineWidth', 1);
    xlabel('Time'); ylabel('Count');
    title(names{i});
    legend('95% CI','50% CI','Median','Data','True','Location','best');
    grid on;
end
sgtitle('MCMC Posterior Predictive Distribution');

%  The bbasic reproduction number R0
R0_chain = chain_burned(:,1) ./ chain_burned(:,2);
R0_true  = beta_true / gamma_true;

fprintf('  R0 true = %.2f\n', R0_true);
fprintf('  R0 posterior mean = %.4f\n', mean(R0_chain));
fprintf('  R0 95%% CI = [%.4f, %.4f]\n', prctile(R0_chain, [2.5, 97.5]));

% The R0 Posterior 
figure('Name',' R0','Position',[100 100 700 450]);
histogram(R0_chain, 80, 'Normalization','pdf', 'FaceColor', [0.3 0.5 0.7]); hold on;
xline(R0_true, 'r--', 'LineWidth', 2);
xline(mean(R0_chain), 'Color', [1 0.6 0], 'LineWidth', 2);
xlabel('R_0 = \beta / \gamma'); ylabel('Density');
title('Posterior Distribution of Basic Reproduction Number R_0');
legend('Posterior', sprintf('True R_0 = %.1f', R0_true), sprintf('Mean = %.2f', mean(R0_chain)));
grid on;

% we do a final Comparison Summary

for i = 1:n_par
    ch = chain_burned(:,i);
    ci_mcmc = prctile(ch, [2.5, 97.5]);
    ci_lsq  = theta_opt(i) + [-1.96, 1.96]*theta_std(i);
    fprintf('  %-8s %-10.4f %-10.4f %-10.4f [%.4f, %.4f]    [%.4f, %.4f]\n', ...
        param_names{i}, theta_true(i), theta_opt(i), mean(ch), ci_lsq, ci_mcmc);
end

%We do a  Bar Comparison 
figure('Name','Comparison','Position',[100 100 800 500]);
mcmc_means = mean(chain_burned);
x = 1:3; w = 0.25;
bar(x-w, theta_true, w, 'FaceColor', 'r', 'FaceAlpha', 0.7); hold on;
bar(x,   theta_opt,  w, 'FaceColor', [0.3 0.5 0.7], 'FaceAlpha', 0.7);
bar(x+w, mcmc_means, w, 'FaceColor', 'g', 'FaceAlpha', 0.7);
for i = 1:3
    ci_l = theta_opt(i) - 1.96*theta_std(i);
    ci_h = theta_opt(i) + 1.96*theta_std(i);
    errorbar(i, theta_opt(i), theta_opt(i)-ci_l, ci_h-theta_opt(i), 'Color',[0.3 0.5 0.7],'LineWidth',2,'CapSize',5);
    ci_mcmc = prctile(chain_burned(:,i), [2.5, 97.5]);
    errorbar(i+w, mcmc_means(i), mcmc_means(i)-ci_mcmc(1), ci_mcmc(2)-mcmc_means(i), 'g','LineWidth',2,'CapSize',5);
end
set(gca, 'XTick', 1:3, 'XTickLabel', {'\beta','\gamma','\phi'});
ylabel('Value'); title('Parameter Estimates with 95% CIs');
legend('True','LSQ','MCMC','Location','best');
grid on;


% The SS Chain 
figure('Name','SS Chain','Position',[100 100 900 350]);
plot(sschain, 'Color', [0.3 0.5 0.7], 'LineWidth', 0.3); hold on;
yline(ss_opt, 'r--', 'LineWidth', 1.5);
xline(burnin, 'Color', [1 0.6 0], 'LineStyle', ':', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('SS');
title('MCMC: Sum of Squares Chain');
legend('SS chain', sprintf('LSQ SS = %.1f', ss_opt), 'Burn-in');
grid on;





