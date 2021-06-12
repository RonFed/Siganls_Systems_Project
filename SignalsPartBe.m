clc;
clear all;
clearvars;
load ('LPF.mat');

%Setup
n_max = 40;
T_s = 1;
num_samples = 2*(n_max/T_s)+1;

%Setup x[n]
n = linspace(-n_max, n_max, num_samples);
t_sampled = T_s*n;
x = 2*cos((3*pi/10)*n).*cos((pi/10)*n);

%y[n]=convolution(x[n],hi[n])
y2 = conv(x,h2,'same');
y3 = conv(x,h3,'same');
y4 = conv(x,h4,'same');
y6 = conv(x,h6,'same');

%Plotting
nexttile
plot(t_sampled, x);
hold on;
plot(t_sampled, y4);
legend('x[n]','y_4[n]');
hold off;
title('x[n] and y_4[n]');
xlabel('n[sec]');

nexttile
plot(t_sampled, x);
hold on;
plot(t_sampled, y6);
legend('x[n]','y_6[n]');
hold off;
title('x[n] and y_6[n]');
xlabel('n[sec]');



