%Group 1

function yvec = sirs_model_vec(theta, data)

[~, y] = ode45(@sirs_ode, data.tspan, data.y0, [], theta, data.N, data.kappa);
yvec = y(:);  
end
