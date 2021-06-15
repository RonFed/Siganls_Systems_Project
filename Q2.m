clc;
clear all;
clearvars;
load ('LPF.mat');

%Question a

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

%%
%Question b

%Setup
n_max = 10000;
T_s = 1;
num_samples = 2*(n_max/T_s)+1;

%Setup x[n]
n = linspace(-n_max,n_max,num_samples);
x = 2*cos((3*pi/10)*n).*cos((pi/10)*n);

%Get X
X = fftshift(fft(x));

%Plotting
w=linspace(-pi,pi,length(X));
plot(w,abs(X));
title('X(e^{j\omega}) absolute frequency plot');
xlabel('\omega[rad/sec]');
ylabel('|X(e^{j\omega})|');

%%
%Question d

%Setup
n_max = 10000;
T_s = 1;
num_samples = 2*(n_max/T_s)+1;

%Setup x[n]
n = linspace(-n_max,n_max,num_samples);
x = 2*cos((3*pi/10)*n).*cos((pi/10)*n);

%y[n]=convolution(x[n],hi[n])
y2 = conv(x,h2);
y3 = conv(x,h3);
y4 = conv(x,h4);
y6 = conv(x,h6);

%DTFT of each y[n]
Y2 = fftshift(fft(y2));
Y3 = fftshift(fft(y3));
Y4 = fftshift(fft(y4));
Y6 = fftshift(fft(y6));

%Omega range
w=linspace(-pi,pi,length(Y2));

%Plotting
nexttile
plot(w,abs(Y2));
title('|Y(e^{j\omega})| plot for h2');
xlabel('\omega[rad/sec]');
ylabel('|Y(e^{j\omega})|');

nexttile
plot(w,abs(Y3));
title('|Y(e^{j\omega})| plot for h3');
xlabel('\omega[rad/sec]');
ylabel('|Y(e^{j\omega})|');

nexttile
plot(w,abs(Y4));
title('|Y(e^{j\omega})| plot for h4');
xlabel('\omega[rad/sec]');
ylabel('|Y(e^{j\omega})|');

nexttile
plot(w,abs(Y6));
title('|Y(e^{j\omega})| plot for h6');
xlabel('\omega[rad/sec]');
ylabel('|Y(e^{j\omega})|');
ylim([0 10000])

%%
%Question e

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