% Test z-transform, s domain -> z domain
clc, clear, close all
% Define the hyperparameter
Sampling_T = 0.003; % Sampling cycle
zeta = 0.18; % damping ratio
wn = 27.78;
K = 771.61;
para1 = 0;
para2 = 1;

s = tf('s');
z=tf('z', Sampling_T);

G = 1/((s)*(s+1)*(s+10));
G_zas = c2d(G, Sampling_T, 'zoh');

% design my control system (PID)
C_as = K * (s+para1) * (s+para2) / s
C_az = c2d(C_as, Sampling_T, 'tustin')

Ls = minreal(G * C_as)    % simply
T_s = minreal(Ls/(1+Ls))  % simply

% print the rlocus
figure();
rlocus(Ls);
title('Ls Root Locus')
hold on; grid on;

Lz = minreal(C_az * G_zas) % simply
T_z = minreal(Lz/(1+Lz))   % simply

% print the rlocus
figure();
rlocus(Ls);
title('Lz Root Locus')
hold on; zgrid(zeta, wn) 

figure();
step(T_s, T_z);
legend('Analog Response', 'Digital Response');
title('Step Response')
grid on;
