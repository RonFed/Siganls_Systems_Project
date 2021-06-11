clc;
clear all;
clearvars;
load ('LPF.mat');

n_max = 10000;
T_s = 1;
num_samples = 2*(n_max/T_s)+1;

n = linspace(-n_max,n_max,num_samples);
t_sampled = T_s*n;
x = 2*cos((3*pi/10)*n).*cos((pi/10)*n);
y2 = conv(x,h2);
Y2 = fftshift(fft(y2));
w=linspace(-pi,pi,length(Y2));
y3 = conv(x,h3);
Y3 = fftshift(fft(y3));
y4 = conv(x,h4);
Y4 = fftshift(fft(y4));
y6 = conv(x,h6);
Y6 = fftshift(fft(y6));

nexttile
plot(w,abs(Y2));
title('Y(e^{j\omega}) absolute frequency plot for h2');
xlabel('\omega[rad/sec]');
ylabel('|Y(e^{j\omega})|');

nexttile
plot(w,abs(Y3));
title('Y(e^{j\omega}) absolute frequency plot for h3');
xlabel('\omega[rad/sec]');
ylabel('|Y(e^{j\omega})|');

nexttile
plot(w,abs(Y4));
title('Y(e^{j\omega}) absolute frequency plot for h4');
xlabel('\omega[rad/sec]');
ylabel('|Y(e^{j\omega})|');

nexttile
plot(w,abs(Y6));
title('Y(e^{j\omega}) absolute frequency plot for h6');
xlabel('\omega[rad/sec]');
ylabel('|Y(e^{j\omega})|');


