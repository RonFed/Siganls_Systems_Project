import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
        
def step(n, n0=0):
    return 1 * (n >= n0)

n = np.arange(-10,10);
h =  pow(0.5,n)*step(n) + pow(0.75,n)*step(n,2)
plt.xlabel('n');
plt.ylabel('h [n]');
plt.title(r'Plot of signal $h[n] = (0.5^n)u[n]+(0.75^n)u[n-2]$');
plt.stem(n, h);
