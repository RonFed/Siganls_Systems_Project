clc;
clear all;
clearvars;
load ('LPF.mat');

%Getting the fft of each h[n]
H2=fftshift(fft(h2));
H3=fftshift(fft(h3));
H4=fftshift(fft(h4));
H6=fftshift(fft(h6));

%Getting the omega range
w2=linspace(-pi,pi,length(H2));
w3=linspace(-pi,pi,length(H3));
w4=linspace(-pi,pi,length(H4));
w6=linspace(-pi,pi,length(H6));

%Plotting
nexttile
plot(w2, abs(H2));
title('Absolute Trnansfer function of h_2');
xlabel('\omega[rad/sec]');
ylabel('|H(j\omega)_2|');

nexttile
plot(w3, abs(H3));
title('Absolute Trnansfer function of h_3');
xlabel('\omega[rad/sec]');
ylabel('|H(j\omega)_3|');

nexttile
plot(w4, abs(H4));
title('Absolute Trnansfer function of h_4');
xlabel('\omega[rad/sec]');
ylabel('|H(j\omega)_4|');

nexttile
plot(w6, abs(H6));
title('Absolute Trnansfer function of h_6');
xlabel('\omega[rad/sec]');
ylabel('|H(j\omega)_6|');
