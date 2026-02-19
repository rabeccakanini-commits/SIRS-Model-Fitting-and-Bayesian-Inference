%Group 1

function ss = sirs_cost(theta, data)

if any(theta <= 0)
    ss = 1e10; 
    return;
end

[~, ymodel] = ode45(@sirs_ode, data.tspan, data.y0, [], theta, data.N, data.kappa);

if size(ymodel,1) ~= size(data.ydata,1)
    ss = 1e10; 
    return;
end

ss = sum(sum((data.ydata - ymodel).^2));
end