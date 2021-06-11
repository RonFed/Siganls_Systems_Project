clc;
clear all;
clearvars;
load ('LPF.mat');

%Getting the fft of each h[n]
N=512;
H2=fftshift(fft(h2,N));
H3=fftshift(fft(h3,N));
H4=fftshift(fft(h4,N));
H6=fftshift(fft(h6,N));

%Getting the omega range
w2=linspace(-pi,pi,length(H2));
w3=linspace(-pi,pi,length(H3));
w4=linspace(-pi,pi,length(H4));
w6=linspace(-pi,pi,length(H6));

%Plotting
nexttile
plot(w2, abs(H2));
title('H_2 absolute frequency plot');
xlabel('\omega');
ylabel('|H[j\omega]_2|');

nexttile
plot(w3, abs(H3));
title('H_3 absolute frequency plot');
xlabel('\omega[rad/sec]');
ylabel('|H[j\omega]_3|');

nexttile
plot(w4, abs(H4));
title('H_4 absolute frequency plot');
xlabel('\omega[rad/sec]');
ylabel('|H[j\omega]_4|');

nexttile
plot(w6, abs(H6));
title('H_6 absolute frequency plot');
xlabel('\omega[rad/sec]');
ylabel('|H[j\omega]_6|');
