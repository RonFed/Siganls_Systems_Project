clc;
clear all;
clearvars;
load ('LPF.mat');

n_max = 10000;
T_s = 1;
num_samples = 2*(n_max/T_s)+1;

n = linspace(-n_max,n_max,num_samples);
x = 2*cos((3*pi/10)*n).*cos((pi/10)*n);
X = fftshift(fft(x));
w=linspace(-pi,pi,length(X));

plot(w,abs(X));
title('X(e^{j\omega}) absolute frequency plot');
xlabel('\omega[rad/sec]');
ylabel('|X(e^{j\omega})|');