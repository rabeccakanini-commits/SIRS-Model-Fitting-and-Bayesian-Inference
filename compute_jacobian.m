%Group 1

function J = compute_jacobian(theta, data)

n_par = length(theta);
f0 = sirs_model_vec(theta, data);
n_obs = length(f0);
J = zeros(n_obs, n_par);

eps_step = 1e-5;

for j = 1:n_par
    theta_p = theta;
    theta_p(j) = theta_p(j) + eps_step;
    fp = sirs_model_vec(theta_p, data);
    J(:,j) = (fp - f0) / eps_step;
end
end
