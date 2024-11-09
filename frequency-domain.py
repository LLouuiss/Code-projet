import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, ifft, fftshift
import math

fontsize = 40
plt.rc('font', size = fontsize)


def read_file(file_name, f_acc = 1024):
    df = pd.read_csv(file_name, sep = ';')
    column = df.columns
    df[column[1:]] = df[column[1:]].applymap(lambda x: str(x).replace(',', '.'))

    time = np.array(df[column[0]], dtype="float64")/f_acc
    acc = np.array(df[column[1]], dtype="float64")
    
    return time, acc

def frequency_domain(t, a):
    y = fft(a)
    n = len(a)
    h = t[1] - t[0]
    x = fftfreq(n, h)
    y = fftshift(y)/n
    x = fftshift(x)
    return x, y

def plot_frequency_domain(x, y, output_name):
    plt.figure(output_name + "_f", figsize=(30,21))
    plt.subplot(211)
    plt.plot(x, np.real(y), 'b', label='Real')
    plt.xlim(x[np.abs(y)>1e-3][0], x[np.abs(y)>1e-3][-1])
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency [Hertz]')
    plt.ylabel('Acceleration [m/s²]')
    plt.subplot(212)
    plt.plot(x, np.imag(y), 'r', label='Imaginary')
    plt.xlim(x[np.abs(y)>1e-3][0], x[np.abs(y)>1e-3][-1])
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency [Hertz]')
    plt.ylabel('Acceleration [m/s²]')
    plt.savefig('figures/frequency_domain/'+output_name+'_f.pdf')

def plot_time_domain(t, a, output_name):
    plt.figure(output_name + "_t", figsize=(30,21))
    plt.plot(t, a, 'b')
    plt.xlim(np.min(t), np.max(t))
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.savefig('figures/frequency_domain/'+output_name+'_t.pdf')

def zero_padding(x,y):
    n = y.size
    h = x[1] - x[0]
    l = math.log2(n)
    z_add = 2**math.ceil(l) - n
    x_new = np.arange(x[0]-z_add//2*h, x[-1] + (z_add - z_add//2 +1)*h, h)
    y_new = np.zeros(2**math.ceil(l))
    y_new[z_add//2-1: n + z_add//2-1] = y
    return x_new, y_new


if __name__ == '__main__':

    file_name = ['Free_vibration_3Masses.csv', "data/Measure_241023_154019.csv", "data/Measure_241023_154607.csv", "data/Measure_241023_155451.csv", "data/Measure_241023_160146.csv", "data/Measure_241106_105003.csv"]
    output_name = ['4-1c']
    f_acc = 1024

    K = 259.55 #N/m
    m = 0.812 #kg
    omega = (K/m)**.5
    
    n = 2**16-100
    h = 1/2**10
    #Point j
    if False:
        t = np.arange(0, n*h, h)
        #Displacement
        u_0 = .05
        u = u_0 * np.cos(omega*t)
        #Velocity
        du = -u_0 * omega * np.sin(omega*t)
        #Acceleration
        ddu = -u_0 * omega**2 * np.cos(omega*t)
        t, ddu = zero_padding(t, ddu)

        x,y = frequency_domain(t, ddu)
        plot_frequency_domain(x,y, "4-1b")
        plot_time_domain(t, ddu, "4-1b")

        
        t, a = read_file(file_name[0], f_acc)
        a = a[int(4.618*f_acc):]
        t = t[int(4.618*f_acc):]
        t, a = zero_padding(t, a)
        x, y = frequency_domain(t, a)
        plot_frequency_domain(x, y, output_name[0])
        plot_time_domain(t, a, output_name[0])

    if False:    
        #Point k

        for xi in [1.39e-3, 2e-2, 5e-2, 1e-1]:
            t = np.arange(0, n*h, h)
            omega_d = omega * math.sqrt(1-xi*xi)
            ddu = xi**2 * omega**2 * np.exp(-xi*omega*t) * (u_0 * np.cos(omega_d*t) + xi*omega*u_0 * np.sin(omega_d*t)/omega_d) - 2 * xi * omega * np.exp(-xi*omega*t) * (-u_0 * omega_d * np.sin(omega_d*t) + xi*omega*u_0 * np.cos(omega_d*t)) + np.exp(-xi*omega*t) * (-u_0 * omega_d**2 * np.cos(omega_d*t) - xi*omega*u_0 * omega_d * np.sin(omega_d*t))
            t, ddu = zero_padding(t, ddu)
            x, y = frequency_domain(t, ddu)
            plot_frequency_domain(x, y, "4-2"+"-xi="+str(xi))
            plot_time_domain(t,ddu, "4-2"+"-xi="+str(xi))

    #Point l
    g = 9.81
    omega__ = np.array([1, .5 , 2, 1.5, .75])
    p_0 = m * .1*g
    xi = 1.39e-3

    if False:
        for i in range(len(omega__)):
            t, a = read_file(file_name[i + 1], f_acc)
            t,a = zero_padding(t,a)
            x,y = frequency_domain(t,a)
            plot_frequency_domain(x,y,"4-3-omega_="+str(omega__[i])+"_exp")
            plot_time_domain(t,a,"4-3-omega_="+str(omega__[i])+"_exp")

            t = np.arange(0, n*h, h)
            phi = math.atan(2*xi*omega__[i]/(1-omega__[i]**2))
            R_d = 1/math.sqrt((1-omega__[i]**2)**2 + (2*xi*omega__[i])**2)
            R_a = omega__[i]**2*R_d
            ddu = p_0/m*R_a*np.sin(omega__[i]*omega*t-phi)
            t, ddu = zero_padding(t, ddu)
            x,y = frequency_domain(t, ddu)
            plot_frequency_domain(x,y,"4-3-omega_="+str(omega__[i])+"_ana")
            plot_time_domain(t,ddu,"4-3-omega_="+str(omega__[i])+"_ana")


    #Point m
    i_last = 25*f_acc
    if True:
        for i in range(len(omega__)):
            t, a = read_file(file_name[i + 1], f_acc)
            t = t[:i_last]
            a = a[:i_last]
            t,a = zero_padding(t,a)
            x,y = frequency_domain(t,a)
            plot_frequency_domain(x,y,"4-4-omega_="+str(omega__[i])+"_exp")
            plot_time_domain(t,a,"4-4-omega_="+str(omega__[i])+"_exp")
    #plt.show()
