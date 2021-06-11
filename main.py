import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.interpolate import interp1d
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
    plt.plot(n, h)
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
    plt.savefig('zeros_poles.png')
    plt.show()


# ~~~~~~~~~~Code for Part C~~~~~~~~~~
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
    omega = np.linspace(-np.pi, np.pi, num=2 * t_max)
    x1 = np.sinc(t / 6)
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
    omega = np.linspace(-1 * np.pi, 1 * np.pi, num=2 * t_max)
    x2 = np.cos(np.pi / 12 * t) + np.sin(np.pi / 6 * t)
    f2 = fftshift(fft(x2))
    plt.xlabel('$\Omega$')
    plt.ylabel('$|X_2(j\Omega$)|')
    plt.title(r'Plot of $X_2(j\Omega$)')
    plt.plot(omega, np.abs(f2))
    plt.savefig('sincfft2.png')
    plt.show()


def Q3_D_DTFT1():
    n_max = 10000
    n = np.arange(-n_max, n_max)
    omega = np.linspace(-np.pi, np.pi, num=2 * n_max)
    x1 = np.sinc(n / 6)
    f1 = fftshift(fft(x1))
    plt.title('Sampled Spectrum of x1[n]')
    plt.xlabel('$\omega[rad/sec] $')
    plt.ylabel('$|X_1(e^{j\omega})|$')
    plt.stem(omega, np.abs(f1))
    plt.savefig('sincfft.png')
    plt.show()


def Q3_D_DTFT2():
    n_max = 10000
    n = np.arange(-n_max, n_max)
    omega = np.linspace(-np.pi, np.pi, num=2 * n_max)
    x2 = np.cos(np.pi / 12 * n) + np.sin(np.pi / 6 * n)
    f2 = fftshift(fft(x2))
    plt.title('Sampled Spectrum of x2[n]')
    plt.xlabel('$\omega[rad/sec] $')
    plt.ylabel('$|X_2(e^{j\omega})|$')
    plt.stem(omega, np.abs(f2))
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

    # Plot X1 and ZOH
    plt.figure()
    plt.plot(t_cont, x_optimal, label="Original")
    plt.plot(t_cont, zoh(t_cont), label="ZOH")
    plt.title(title)
    plt.xlabel("Time [sec]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t_cont, x_optimal, label="Original")
    plt.plot(t_cont, foh(t_cont), label="FOH")
    plt.title(title)
    plt.xlabel("Time [sec]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def ZOH_FOH_SPECT(func, T_s):
    n_max = 10000
    n = np.arange(-n_max, n_max)
    t_sampled = T_s*n
    t_cont = np.linspace(-n_max*T_s, n_max*T_s, num=len(t_sampled) * 10)
    omega = np.linspace(-np.pi, np.pi, num=2 * n_max)
    h_zoh = zoh_filter(t_sampled, T_s)
    h_perfect = perfect_filter(omega, T_s)
    x_sampled = func(t_sampled)
    f_sampled = fftshift(fft(x_sampled))
    # x_rec_zoh = ifft(f_sampled*h_zoh)
    # x_rec_perfect = ifft(f_sampled*h_perfect)
    # plt.plot(t_sampled[9900:10100],x_rec_zoh[9900:10100])
    # plt.plot(t_sampled[9900:10100], x_rec_perfect[9900:10100])
    y = np.convolve(x_sampled, h_zoh)
    t_present = [t for t in range(len(t_sampled)) if abs(t_sampled[t]) <=10]
    plt.plot(t_sampled[t_present],y[t_present])
    plt.plot(t_sampled[t_present],x_sampled[t_present])
    plt.plot(t_sampled[t_present],h_zoh[t_present])



def perfect_filter(omega, T_s):
    return T_s * (np.abs(omega) <= np.pi / T_s)


def zoh_filter(t, T_s):
    return 1 * (np.abs(t) <= T_s/2)

# def zoh_filter(omega, T_s):
#     return T_s * np.sinc((T_s * omega) / (2 * np.pi))


def foh_filter(omega, T_s):
    return T_s * (np.sinc((T_s * omega) / (2 * np.pi))) ** 2


def RunQ3_E():
    T_s = 1
    Q3_E_ZOH_FOH(T_s, lambda t: np.sinc(t / 6),
                 "$x_1(t)=sinc(\\frac{t}{6}),  T_s = $" + str(T_s) + " seconds", "$x_1(t)$")
    Q3_E_ZOH_FOH(T_s, lambda t: np.cos((np.pi / 12) * t) + np.sin((np.pi / 6) * t),
                 "$x_2(t)=cos(\\frac{\pi}{12}t) + sin(\\frac{\pi}{6}t),  T_s = $" + str(T_s) + " seconds", "$x_2(t)$")


# Q3_D_DTFT1()
# Q3_D_DTFT2()
# RunQ3_E()
ZOH_FOH_SPECT(lambda t: np.cos((np.pi / 12) * t) + np.sin((np.pi / 6) * t), 0.1)