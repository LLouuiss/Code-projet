import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, ifft, fftshift
import math

#Initialize font size for the plots
fontsize = 40
plt.rc('font', size = fontsize)

"""
Read a csv file and return the time and acceleration
Input:
    file_name: string name of the file
    f_acc: int frequency of the sampling of the acceleration
Output:
    time: np.array time
    acc: np.array acceleration
"""
def read_file(file_name, f_acc = 1024):
    df = pd.read_csv(file_name, sep = ';')
    column = df.columns
    df[column[1:]] = df[column[1:]].applymap(lambda x: str(x).replace(',', '.'))

    time = np.array(df[column[0]], dtype="float64")/f_acc
    acc = np.array(df[column[1]], dtype="float64")
    
    return time, acc

"""
Transform a function from time domain to frequency domain by using the Fast Fourier Transform
Input:
    t: np.array time
    a: np.array acceleration in time domain
    zero_padding: boolean to use zero padding
Output:
    x: np.array frequency
    y: np.array acceleration in frequency domain
"""
def frequency_domain(t, a, zero_padding = False):
    if zero_padding:
        n = 2**math.ceil(math.log2(len(a)))
    else:
        n = len(a)
    y = fft(a, n)
    h = t[1] - t[0]
    x = fftfreq(n, h)
    y = fftshift(y)/n
    x = fftshift(x)
    return x, y

"""
Plot a function in the frequency domain
Input:
    x: np.array frequency
    y: np.array acceleration in frequency domain
    output_name: string name of the output file
    boundary: float remove the extremities of the graph where abs(y) is smaller than boundary
Output:
    None
"""
def plot_frequency_domain(x, y, output_name, boundary = 2e-2):
    plt.figure(output_name + "_f", figsize=(30,21))
    #Real part
    plt.subplot(211)
    plt.plot(x, np.real(y), 'b', label='Real')
    try:
        plt.xlim(x[np.abs(y)>boundary][0], x[np.abs(y)>boundary][-1])
    except IndexError:
        plt.xlim(x[3*x.size//8], x[5*x.size//8])
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency [Hertz]')
    plt.ylabel('Acceleration [m/s²]')
    #Imaginary part
    plt.subplot(212)
    plt.plot(x, np.imag(y), 'r', label='Imaginary')
    try:
        plt.xlim(x[np.abs(y)>boundary][0], x[np.abs(y)>boundary][-1])
    except IndexError:
        plt.xlim(x[3*x.size//8], x[5*x.size//8])
    plt.grid(True)
    plt.legend()
    plt.xlabel('Frequency [Hertz]')
    plt.ylabel('Acceleration [m/s²]')
    plt.savefig('figures/frequency_domain/'+output_name+'_f.pdf')

"""
Plot a function in the time domain
Input:
    t: np.array time
    a: np.array acceleration in time domain
    output_name: string name of the output file
Output:
    None
"""
def plot_time_domain(t, a, output_name):
    plt.figure(output_name + "_t", figsize=(30,21))
    plt.plot(t, a, 'b')
    plt.xlim(np.min(t), np.max(t))
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.savefig('figures/frequency_domain/'+output_name+'_t.pdf')




if __name__ == '__main__':

    #file with data
    file_name = ['Free_vibration_3Masses.csv', "data/Measure_241023_154019.csv", "data/Measure_241023_154607.csv", "data/Measure_241023_155451.csv", "data/Measure_241023_160146.csv", "data/Measure_241106_105003.csv"]
    #file where the output will be saved
    output_name = ['4-1c']
    #frequency of the sampling of the acceleration
    f_acc = 1024

    #Data
    #Stiffness
    K = 259.55 #N/m
    #Mass
    m = 0.812 #kg
    #Natural Cyclic Frequency
    omega = (K/m)**.5
    f = omega/(2*math.pi)
    print(f"Natural frequency: {f:.2f} Hertz")
    
    #Time
    n = 2**16-100
    h = 1/2**10
    t = np.arange(0, n*h, h)
    #Initial displacement for the free vibration
    u_0 = .05

    #Point j
    if False:
        #Displacement
        u = u_0 * np.cos(omega*t)
        #Velocity
        du = -u_0 * omega * np.sin(omega*t)
        #Acceleration
        ddu = -u_0 * omega**2 * np.cos(omega*t)
        #Plot Graphs for the analytical part
        x,y = frequency_domain(t, ddu, True)
        plot_frequency_domain(x,y, "4-1b")
        plot_time_domain(t, ddu, "4-1b")

        #Plot Graphs for the experimental part
        t, a = read_file(file_name[0], f_acc)
        a = a[int(4.618*f_acc):]
        t = t[int(4.618*f_acc):]
        x, y = frequency_domain(t, a, True)
        plot_frequency_domain(x, y, output_name[0])
        plot_time_domain(t, a, output_name[0])
    boundary = [2e-2, 1e-2, 1e-2, 1e-2]
    #Point k
    if False:    
        #Values of xi
        for i, xi in enumerate([1.39e-3, 2e-2, 5e-2, 1e-1]):
            print(f"xi: {xi:.2e} => fd = {f*(1-xi**2)**.5:.2f} Hertz")
            #Time
            t = np.arange(0, n*h, h)
            #damped natural frequency
            omega_d = omega * math.sqrt(1-xi*xi)
            #Analytical Acceleration
            ddu = xi**2 * omega**2 * np.exp(-xi*omega*t) * (u_0 * np.cos(omega_d*t) + xi*omega*u_0 * np.sin(omega_d*t)/omega_d) - 2 * xi * omega * np.exp(-xi*omega*t) * (-u_0 * omega_d * np.sin(omega_d*t) + xi*omega*u_0 * np.cos(omega_d*t)) + np.exp(-xi*omega*t) * (-u_0 * omega_d**2 * np.cos(omega_d*t) - xi*omega*u_0 * omega_d * np.sin(omega_d*t))
            
            x, y = frequency_domain(t, ddu, True)
            plot_frequency_domain(x, y, "4-2"+"-xi="+str(xi), boundary[i])
            plot_time_domain(t,ddu, "4-2"+"-xi="+str(xi))

    #Point l
    #Gravity
    g = 9.81 #m/s²
    #Constant of multiplication for the frequency
    omega__ = np.array([1, .5 , 2, 1.5, .75])
    #Amplitude of the force
    p_0 = m * .1*g #N
    #Damping ratio
    xi = 1.39e-3
    #Index of the start and end of the usable data
    i_start = np.array(f_acc * np.array([4, 20, 5, 7, 7]), dtype=int)
    i_end = np.array(f_acc * np.array([65, 60, 60, 60, 60]), dtype=int)
    if False:
        for i in range(len(omega__)):
            #Experimental part
            t, a = read_file(file_name[i + 1], f_acc)
            t = t[i_start[i]:i_end[i]]
            a = a[i_start[i]:i_end[i]]
            x,y = frequency_domain(t,a, True)
            plot_frequency_domain(x,y,"4-3-omega_="+str(omega__[i])+"_exp", 5e-2)
            plot_time_domain(t,a,"4-3-omega_="+str(omega__[i])+"_exp")

            #Analytical part
            t = np.arange(0, n*h, h)
            phi = math.atan(2*xi*omega__[i]/(1-omega__[i]**2))
            R_d = 1/math.sqrt((1-omega__[i]**2)**2 + (2*xi*omega__[i])**2)
            R_a = omega__[i]**2*R_d
            ddu = - p_0/m*R_a*np.sin(omega__[i]*omega*t-phi)
            x,y = frequency_domain(t, ddu, True)
            plot_frequency_domain(x,y,"4-3-omega_="+str(omega__[i])+"_ana", 5e-3)
            plot_time_domain(t,ddu,"4-3-omega_="+str(omega__[i])+"_ana")


    #Point m
    #Window of the data
    di = 25*f_acc
    if True:
        for i in range(len(omega__)):
            t, a = read_file(file_name[i + 1], f_acc)
            t = t[i_start[i]:i_start[i] + di]
            a = a[i_start[i]:i_start[i] + di]
            x,y = frequency_domain(t,a, True)
            plot_frequency_domain(x,y,"4-4-omega_="+str(omega__[i])+"_exp")
            plot_time_domain(t,a,"4-4-omega_="+str(omega__[i])+"_exp")
    #plt.show()
