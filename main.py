import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.interpolate import interp1d
import numpy as np
from scipy import signal
import control


#~~~~~~~~~~Code for Part A~~~~~~~~~~
def step(n, n0=0):
    return 1 * (n >= n0)


def Q1_D():
    n = np.arange(-10, 10)
    h = pow(0.5, n) * step(n) + pow(0.75, n) * step(n, 2)
    plt.xlabel('n')
    plt.ylabel('h [n]')
    plt.title(r'Impulse response of $x[n]=(0.5^n)u[n]+(0.75^n)u[n-2]$')
    plt.stem(n, h)
    plt.show()


def Q1_E_Transfer_Function():
    h = signal.TransferFunction([1, -0.75, 9 / 16, -9 / 32], [1, -5 / 4, 3 / 8, 0])
    print("Zeros: ",h.zeros)
    print("Poles: ",h.poles)
    w, mag, phase = signal.bode(h)
    plt.figure()
    plt.semilogx(w, mag)  # Bode magnitude plot
    plt.figure()
    plt.semilogx(w, phase)  # Bode phase plot
    plt.show()
    

def Q1_F_poles_zeros():
    h2 = control.TransferFunction([1, -0.75, 9 / 16, -9 / 32], [1, -5 / 4, 3 / 8, 0])
    control.pzmap(h2, plot=True)
    plt.savefig('zeros_poles.png')
    plt.show()
    


#~~~~~~~~~~Code for Part C~~~~~~~~~~
def delta(n, n0=0):
    return 1 * (np.abs(n - n0) <= 0.01)


def Q3_A_From_Our_Calc():
    x = np.linspace(-np.pi, np.pi, 1000)
    y = 6 * (step(x, -np.pi / 6) - step(x, np.pi / 6))
    plt.xlabel('$\Omega$')
    plt.ylabel('$|X_1(j\Omega$)|')
    plt.title(r'Plot of $X_1(j\Omega$)')
    plt.plot(x, y)
    plt.show()

    x = np.linspace(-np.pi, np.pi, 1000)
    y = np.pi * (delta(x, -np.pi / 12) + delta(x, np.pi / 12) + delta(x, -np.pi / 6) + delta(x, np.pi / 6))
    plt.xlabel('$\Omega$')
    plt.ylabel('$|X_2(j\Omega$)|')
    plt.title(r'Plot of $X_2(j\Omega$)')
    plt.plot(x, y)
    plt.show()
    

def Q3_A_FFT1():
    t_max = 100;
    t = np.arange(-t_max, t_max)
    omega = np.linspace(-np.pi, np.pi, num=2*t_max)
    x1 = np.sinc(t/6)
    f1 = fftshift(fft(x1))
    plt.xlabel('$\Omega$')
    plt.ylabel('$|X_2(j\Omega$)|')
    plt.title(r'Plot of $X_2(j\Omega$)')
    plt.plot(omega, np.abs(f1))
    plt.savefig('sincfft.png')
    plt.show()
  

def Q3_A_FFT2():
    t_max = 100;
    t = np.arange(-t_max, t_max)
    omega = np.linspace(-1 * np.pi, 1 * np.pi, num=2*t_max)
    x2 = np.cos(np.pi / 12 * t) + np.sin(np.pi / 6 * t)
    f2 = fftshift(fft(x2))
    plt.xlabel('$\Omega$')
    plt.ylabel('$|X_2(j\Omega$)|')
    plt.title(r'Plot of $X_2(j\Omega$)')
    plt.plot(omega, np.abs(f2))
    plt.savefig('sincfft2.png')
    plt.show()
    
    
def Q3_D_DTFT(T_s):
    n_max = 10000
    n = np.arange(-n_max, n_max)
    t_sampled = n * T_s
    omega = np.linspace(-np.pi, np.pi, num=2*n_max)
    x1 = np.sinc(t_sampled / 6)
    f1 = fftshift(fft(x1))
    plt.figure()
    plt.title('Sampled Spectrum of x1[n]')
    plt.xlabel('$\omega[rad/sec] $')
    plt.ylabel('$|X_1(e^{j\omega})|$')
    plt.plot(omega, np.abs(f1))
    plt.savefig('sincfft.png')
    plt.show()

    plt.figure()
    x2 = np.cos(np.pi / 12 * t_sampled) + np.sin(np.pi / 6 * t_sampled)
    f2 = fftshift(fft(x2))
    plt.title('Sampled Spectrum of x2[n]')
    plt.xlabel('$\omega[rad/sec] $')
    plt.ylabel('$|X_2(e^{j\omega})|$')
    plt.plot(omega, np.abs(f2))
    plt.savefig('sincfft2.png')
    plt.show()




def Q3_E_ZOH_FOH(T_s, func, title, ylabel):
    abs_max_time = 20
    num_of_samples = (2 * abs_max_time) // T_s + 1

    t_sampled = np.linspace(-abs_max_time, abs_max_time, num=num_of_samples)
    t_cont = np.linspace(-abs_max_time, abs_max_time, num=num_of_samples * 100)

    x_sampled = func(t_sampled)
    zoh = interp1d(t_sampled, x_sampled, kind='previous')
    foh = interp1d(t_sampled, x_sampled)

    x_optimal = func(t_cont)
    
    #Plot ZOH
    plt.figure()
    plt.plot(t_cont, x_optimal, label="Original")
    plt.plot(t_cont, zoh(t_cont), label="ZOH")
    plt.title(title)
    plt.xlabel("Time [sec]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
    #Plot FOH
    plt.figure()
    plt.plot(t_cont, x_optimal, label="Original")
    plt.plot(t_cont, foh(t_cont), label="FOH")
    plt.title(title)
    plt.xlabel("Time [sec]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def RunQ3_E():
    T_s = 9
    Q3_E_ZOH_FOH(T_s, lambda t: np.sinc(t / 6),
            "$x_1(t)=sinc(\\frac{t}{6}),  T_s = $" + str(T_s) + " seconds", "$x_1(t)$")
    Q3_E_ZOH_FOH(T_s, lambda t: np.cos((np.pi / 12) * t) + np.sin((np.pi / 6) * t),
            "$x_2(t)=cos(\\frac{\pi}{12}t) + sin(\\frac{\pi}{6}t),  T_s = $" + str(T_s) + " seconds", "$x_2(t)$")    


# Q1_D()
Q3_D_DTFT(9)
#Q3_D_DTFT1()
#Q3_D_DTFT2()
RunQ3_E()