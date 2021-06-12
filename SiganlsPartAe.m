clc;
clear;

syms n;
sympref('HeavisideAtOrigin', 1);
f = 0.5^n*heaviside(n) + 0.75^n*heaviside(n-2);
z_f=ztrans(f);
simplifyFraction(z_f)