clc;
clear all;
clearvars;

T_s = 5;
t_max = 20;
num_samples = (2*t_max)/T_s + 1;
t_sampled = linspace(-t_max,t_max,num_samples);
t_cont = linspace(-t_max,t_max,num_samples * 100);

zoh_time = zeros(1,length(t_sampled));
for n = 1:length(t_sampled)
   if abs(t_sampled(n)) <= T_s/2 + 1
       zoh_time(n) = 1/T_s;
   end
end

foh_time = zeros(1,length(t_cont));
for n = 1:length(t_cont)
   if abs(t_cont(n)) <= T_s
       foh_time(n) = 1-abs(t_cont(n))/T_s;
   end
end

ideal_time = sinc(t_cont);
x1 = sinc(t_sampled/6);
zoh_x1 = conv(x1,zoh_time, 'same');

hold on
plot(t_sampled, x1);
plot(t_sampled, zoh_x1);
