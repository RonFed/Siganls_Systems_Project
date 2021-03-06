clc;
clear all;
clearvars;

T_s = 1;
t_max = 15;
num_samples = (2*t_max)/T_s + 1;
t_sampled = linspace(-t_max,t_max,num_samples);
t_cont = linspace(-t_max,t_max,num_samples * 100);
x1 = [-t_max:T_s:t_max ; sinc(t_sampled/6)]';
y_zoh = pulstran(t_cont,x1,@rectpuls,T_s);
y_foh = pulstran(t_cont,x1,@tripuls,2*T_s);
fnx = @(t) sinc(t/T_s);
y_ideal = pulstran(t_cont,x1,fnx(t_cont));

hold on;
%plot(t_cont,y_zoh);
%plot(t_cont,y_foh);
plot(t_cont,y_ideal,'r');
%plot(t_cont,sinc(t_cont/6),'k');
%%
zoh_time = zeros(1,length(t_cont));
for n = 1:length(t_cont)
   if and(t_cont(n) >= 0, t_cont(n)<= T_s)
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
%plot(t_sampled, zoh_time);
%plot(t_sampled, zoh_x1);
