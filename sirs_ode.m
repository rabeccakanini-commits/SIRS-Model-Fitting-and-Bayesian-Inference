%Group 1

function dy = sirs_ode(t, y, theta, N, kappa)

S = y(1);  I = y(2);  R = y(3);
beta = theta(1);  gamma = theta(2);  phi = theta(3);

dS = kappa*N - beta*S*I/N + phi*R - kappa*S;
dI = beta*S*I/N - gamma*I - kappa*I;
dR = gamma*I - phi*R - kappa*R;

dy = [dS; dI; dR];
end
