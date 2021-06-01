import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
        
def step(n, n0=0):
    return 1 * (n >= n0)

def delta(n, n0=0):
    return 1 * (np.abs(n-n0)<=0.01)
    
    
def q1_D():
    n = np.arange(-10,10)
    h =  pow(0.5,n)*step(n) + pow(0.75,n)*step(n,2)
    plt.xlabel('n')
    plt.ylabel('h [n]')
    plt.title(r'Plot of signal $h[n] = (0.5^n)u[n]+(0.75^n)u[n-2]$')
    plt.stem(n, h)
    plt.show()
    
def q3_A():
    x = np.linspace(-np.pi,np.pi,1000)
    y = 6*(step(x,-np.pi/6) - step(x,np.pi/6))
    plt.xlabel('$\Omega$')
    plt.ylabel('$|X_1(j\Omega$)|')
    plt.title(r'Plot of $X_1(j\Omega$)')
    plt.plot(x, y)
    plt.show()
    
    x = np.linspace(-np.pi,np.pi,1000)
    y = np.pi*(delta(x,-np.pi/12)+delta(x,np.pi/12)+delta(x,-np.pi/6)+delta(x,np.pi/6))
    plt.xlabel('$\Omega$')
    plt.ylabel('$|X_2(j\Omega$)|')
    plt.title(r'Plot of $X_2(j\Omega$)')
    plt.plot(x, y)
    plt.show()

q1_D()
q3_A()
