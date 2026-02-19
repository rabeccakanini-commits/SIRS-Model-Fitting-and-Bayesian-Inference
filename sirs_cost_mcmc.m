%Group 1

function ss = sirs_cost_mcmc(theta, data)

tspan = data.xdata(:)';
ydata = data.ydata;

if any(theta <= 0)
    ss = 1e10; 
    return;
end

[~, ymodel] = ode45(@sirs_ode, tspan, data.y0, [], theta, data.N, data.kappa);

if size(ymodel,1) ~= size(ydata,1)
    ss = 1e10; 
    return;
end

ss = sum(sum((ydata - ymodel).^2));
end