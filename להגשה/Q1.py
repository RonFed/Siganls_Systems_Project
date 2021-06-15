import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import control


# ~~~~~~~~~~Code for Part A~~~~~~~~~~
def step(n, n0=0):
    return 1 * (n >= n0)


def Q1_D():
    n = np.arange(-10, 10)
    h = pow(0.5, n) * step(n) + pow(0.75, n) * step(n, 2)
    plt.xlabel('n')
    plt.ylabel('h [n]')
    plt.title(r'Impulse response of $x[n]=(0.5^n)u[n]+(0.75^n)u[n-2]$')
    plt.stem(n, h)
    plt.savefig('Q1D.png')
    plt.show()


def Q1_E_Transfer_Function():
    h = signal.TransferFunction([1, -0.75, 9 / 16, -9 / 32], [1, -5 / 4, 3 / 8, 0])
    print("Zeros: ", h.zeros)
    print("Poles: ", h.poles)
    w, mag, phase = signal.bode(h)
    plt.figure()
    plt.semilogx(w, mag)  # Bode magnitude plot
    plt.figure()
    plt.semilogx(w, phase)  # Bode phase plot
    plt.show()


def Q1_F_poles_zeros():
    h2 = control.TransferFunction([1, -0.75, 9 / 16, -9 / 32], [1, -5 / 4, 3 / 8, 0])
    control.pzmap(h2, plot=True)
    plt.savefig('Q1F_zeros_poles.png')
    plt.show()

