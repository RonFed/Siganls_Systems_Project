import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.interpolate import interp1d
import numpy as np


# ~~~~~~~~~~Code for Part C~~~~~~~~~~
def Q3_A_FFT():
    t_max = 100;
    t = np.arange(-t_max, t_max)
    omega = np.linspace(-np.pi, np.pi, num=2 * t_max)

    x1 = np.sinc(t / 6)
    x2 = np.cos((np.pi / 12) * t) + np.sin((np.pi / 6) * t)

    f1 = fftshift(fft(x1))
    f2 = fftshift(fft(x2))

    plt.figure()
    plt.xlabel('$\Omega[rad/sec]$')
    plt.ylabel('$|X_1(j\Omega$)|')
    plt.title(r'Plot of $X_1(j\Omega$)')
    plt.plot(omega, np.abs(f1))
    plt.savefig('Q3A_FFT1.png')
    plt.show()

    plt.figure()
    plt.xlabel('$\Omega[rad/sec]$')
    plt.ylabel('$|X_2(j\Omega$)|')
    plt.title(r'Plot of $X_2(j\Omega$)')
    plt.plot(omega, np.abs(f2))
    plt.savefig('Q3A_FFT2.png')
    plt.show()


def Q3_D_DTFT(T_s=1):
    n_max = 10000
    n = np.arange(-n_max, n_max)
    t_sampled = n * T_s
    omega = np.linspace(-np.pi, np.pi, num=2 * n_max)
    x1 = np.sinc(t_sampled / 6)
    f1 = fftshift(fft(x1))
    plt.figure()
    plt.title('Sampled Spectrum of x1[n]')
    plt.xlabel('$\omega[rad/sec] $')
    plt.ylabel('$|X_1(e^{j\omega})|$')
    plt.plot(omega, np.abs(f1))
    plt.savefig('Q3D_DTFT1_ ' + str(T_s) + '.png')
    plt.show()

    plt.figure()
    x2 = np.cos((np.pi / 12) * t_sampled) + np.sin((np.pi / 6) * t_sampled)
    f2 = fftshift(fft(x2))
    plt.title('Sampled Spectrum of x2[n]')
    plt.xlabel('$\omega[rad/sec] $')
    plt.ylabel('$|X_2(e^{j\omega})|$')
    plt.plot(omega, np.abs(f2))
    plt.savefig('Q3D_DTFT2_ ' + str(T_s) + '.png')
    plt.show()


def Q3_E_ZOH_FOH(T_s, func, func_name, title, ylabel):
    abs_max_time = 40
    num_of_samples = (2 * abs_max_time) // T_s + 1

    t_sampled = np.linspace(-abs_max_time, abs_max_time, num=num_of_samples)
    t_cont = np.linspace(-abs_max_time, abs_max_time, num=num_of_samples * 100)

    x_original = func(t_cont)

    x_sampled = func(t_sampled)
    zoh = interp1d(t_sampled, x_sampled, kind='previous')
    foh = interp1d(t_sampled, x_sampled)
    ideal = np.convolve(x_sampled, np.sinc(t_sampled / T_s), 'same')

    # Plot ZOH
    plt.figure()
    plt.plot(t_cont, x_original, label="Original")
    plt.plot(t_cont, zoh(t_cont), label="ZOH")
    plt.title(title)
    plt.xlabel("Time [sec]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('Q3E_ ' + func_name + '_ZOH_ ' + str(T_s) + '.png')
    plt.show()

    # Plot FOH
    plt.figure()
    plt.plot(t_cont, x_original, label="Original")
    plt.plot(t_cont, foh(t_cont), label="FOH")
    plt.title(title)
    plt.xlabel("Time [sec]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('Q3E_ ' + func_name + '_FOH_ ' + str(T_s) + '.png')
    plt.show()

    # Plot Ideal
    plt.figure()
    plt.plot(t_cont, x_original, label="Original")
    plt.plot(t_sampled, ideal, label="Ideal")
    plt.title(title)
    plt.xlabel("Time [sec]")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('Q3E_ ' + func_name + '_Ideal_ ' + str(T_s) + '.png')
    plt.show()


# ~~~~~~~~~~Run codes~~~~~~~~~~
def RunQ3_E(T_s=1):
    Q3_E_ZOH_FOH(T_s, lambda t: np.sinc(t / 6), 'X1',
                 "$x_1(t)=sinc(\\frac{t}{6}),  T_s = $" + str(T_s) + " seconds", "$x_1(t)$")
    Q3_E_ZOH_FOH(T_s, lambda t: np.cos((np.pi / 12) * t) + np.sin((np.pi / 6) * t), 'X2',
                 "$x_2(t)=cos(\\frac{\pi}{12}t) + sin(\\frac{\pi}{6}t),  T_s = $" + str(T_s) + " seconds", "$x_2(t)$")

RunQ3_E(9)