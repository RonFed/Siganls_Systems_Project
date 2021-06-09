import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.interpolate import interp1d
import numpy as np
from scipy import signal
import control


def zoh_foh(T_s, func, title, ylabel):
    T_s_optimal = T_s / 100
    max_time = 10
    max_n = int(1e4)
    n = np.arange(-max_n, max_n + 1)

    t = n * T_s
    t_present = np.array([time for time in t if abs(time) <= max_time / 2])
    x_present = func(t_present)
    zoh = interp1d(t_present, x_present, kind='previous')
    foh = interp1d(t_present, x_present)

    t_optimal = n * T_s_optimal
    t_present_optimal = np.array([time for time in t_optimal if abs(time) <= max_time / 2])
    x_present_optimal = func(t_present_optimal)

    plt.plot(t_present_optimal, zoh(t_present_optimal), label="ZOH")
    plt.plot(t_present_optimal, foh(t_present_optimal), label="FOH")
    plt.plot(t_present_optimal, x_present_optimal, label="Optimal")
    plt.xlabel("Time [sec]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def fft_sampled1():
    n = np.arange(-10000, 10000)
    omega = np.linspace(-1 * np.pi, 1 * np.pi, num=20000)
    x1 = np.sinc(n / 6)
    f1 = fftshift(fft(x1))
    plt.title('Sampled Spectrum of x1[n]')
    plt.xlabel('$\omega $')
    plt.ylabel('$|X_1(e^{j\omega})|$')
    plt.plot(omega, np.abs(f1))
    plt.savefig('sincfft.png')
    plt.show()


def fft_sampled2():
    n = np.arange(-10000, 10000)
    omega = np.linspace(-1 * np.pi, 1 * np.pi, num=20000)
    x2 = np.cos(np.pi / 12 * n) + np.sin(np.pi / 6 * n)
    f2 = fftshift(fft(x2))
    plt.title('Sampled Spectrum of x2[n]')
    plt.xlabel('$\omega $')
    plt.ylabel('$|X_2(e^{j\omega})|$')
    plt.plot(omega, np.abs(f2))
    plt.savefig('sincfft2.png')
    plt.show()


def poles_zeros():
    h2 = control.TransferFunction([1, -0.75, 9 / 16, -9 / 32], [1, -5 / 4, 3 / 8, 0])
    control.pzmap(h2, plot=True)
    plt.savefig('zeros_poles.png')
    plt.show()


def transfer_function_1():
    h = signal.TransferFunction([1, -0.75, 9 / 16, -9 / 32], [1, -5 / 4, 3 / 8, 0])
    print(h.zeros)
    print(h.poles)
    w, mag, phase = signal.bode(h)
    plt.figure()
    plt.semilogx(w, mag)  # Bode magnitude plot

    # plt.figure()
    # plt.semilogx(w, phase)  # Bode phase plot
    plt.show()


def step(n, n0=0):
    return 1 * (n >= n0)


def delta(n, n0=0):
    return 1 * (np.abs(n - n0) <= 0.01)


def q1_D():
    n = np.arange(-10, 10)
    h = pow(0.5, n) * step(n) + pow(0.75, n) * step(n, 2)
    plt.xlabel('n')
    plt.ylabel('h [n]')
    plt.title(r'Plot of signal $h[n] = (0.5^n)u[n]+(0.75^n)u[n-2]$')
    plt.stem(n, h)
    plt.show()

    N = 100
    T = 1.0 / 1000.0
    n = fftfreq(N, T)[:N // 2]
    h = pow(0.5, n) * step(n) + pow(0.75, n) * step(n, 2)


def q3_A():
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


zoh_foh(1, lambda t: np.sinc(t / 6),
        "$x_1(t)=sinc(\\frac{t}{6}),  T_s = $" + str(1) + " seconds", "$x_1(t)$")
zoh_foh(1, lambda t: np.cos((np.pi / 12) * t) + np.sin((np.pi / 6) * t),
        "$x_2(t)=cos(\\frac{\pi}{12}t) + sin(\\frac{\pi}{6}t),  T_s = $" + str(1) + " seconds", "$x_2(t)$")
# fft_sampled2()
# transfer_function_1()
# poles_zeros()
# q1_D()
# q3_A()
