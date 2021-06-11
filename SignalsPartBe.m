clc;
clear all;
clearvars;
load ('LPF.mat');

%Setup
n_max = 10000;
T_s = 1;
num_samples = 2*(n_max/T_s)+1;

%Setup x[n]
n = linspace(-n_max,n_max);
t_sampled = T_s*n;
x = 2*cos((3*pi/10)*n).*cos((pi/10)*n);

%y[n]=convolution(x[n],hi[n])
y2 = conv(x,h2);
y3 = conv(x,h3);
y4 = conv(x,h4);
y6 = conv(x,h6);

%Plotting
x_max = 10;
Xaxis = linspace(-x_max,x_max);
% nexttile
% plot(n,x,'b*-');
% hold on;
% plot(n,y2, 'r*-');
% title('x[n] and y_2[n]');
% xlabel('n[sec]');

nexttile
plot(Xaxis,x);
hold on;
%plot(Xaxis,y6);
hold off;
title('x[n] and y_3[n]');
xlabel('n[sec]');




