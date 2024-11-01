import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

def plot_subplot(t, u, Time_range, nrows, ncols, index, label, color = 'b', max_y = None):
    plt.subplot(nrows, ncols, index)
    plt.plot(t, u, color)
    plt.ylabel(label)
    plt.grid(True)
    plt.xlim(Time_range)
    if max_y != None :
        plt.yticks(np.linspace(-max_y, max_y, 5))

def interpolate(t, u, t_new):
    if t_new <= t[0]:
        return u[0]
    elif t_new >= t[-1]:
        return u[-1]
    else:
        for j in range(t.shape[0] - 1):
            if t[j] <= t_new and t_new <= t[j+1]:
                return u[j] + (u[j+1] - u[j])/(t[j+1] - t[j])*(t_new - t[j])
    return 0

def Runge_Kutta_4(f, U_0, h, n):
    U = np.zeros((n, len(U_0)), dtype='float64')
    t = np.arange(0, n*h, h , dtype='float64')
    U[0] = U_0
    for i in range(1, n):
        k1 = f(t[i-1], U[i-1])
        k2 = f(t[i-1] + h/2, U[i-1] + h/2*k1)
        k3 = f(t[i-1] + h/2, U[i-1] + h/2*k2)
        k4 = f(t[i-1] + h, U[i-1] + h*k3)
        U[i] = U[i-1] + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return t, U

def Adams_Bashforth_2(f, U_0, h, n, t) :
    U = np.zeros((n, len(U_0)), dtype='float64')
    U[0] = U_0
    for i in range(1, n):
        U[i] = U[i-1] + h/2*(f(t[i-1], U[i-1]) + f(t[i], U[i]))
    return t, U

def Adams_Bashforth_4(f, U_0, h, n, t) :
    U = np.zeros((n, len(U_0)), dtype='float64')
    U[0] = U_0
    U[1] = U[0] + h/2*(f(t[0], U[0]) + f(t[1], U[1]))
    U[2] = U[1] + h/12*(-f(t[0],U[0]) + 8*f(t[1], U[1]) + 5*f(t[2], U[2]))
    for i in range(3, n):
        U[i] = U[i-1] + h/24*(f(t[i-3], U[i-3]) - 5*f(t[i-2], U[i-2]) + 19*f(t[i-1], U[i-1]) + 9*f(t[i], U[i]))
    return t, U

if __name__ == '__main__':

    p1_2 = False
    p1_3 = True


    #Data
    E = 210e9 #Pa
    I = 7.158e-12 #m^4
    L = .275 #m
    m = .711 #kg

    #Stiffness
    K = 3*E*I/(L**3)
    #Natural Cyclic Frequency
    omega = (K/m)**.5
    #Natural Period
    T = 2*np.pi/omega

    #Time
    Time_range = (0, 10)
    nb = 100
    t = np.arange(Time_range[0], Time_range[1] + T/nb, T/nb)

    #Displacement
    u_0 = .05
    u = u_0 * np.cos(omega*t)
    #Velocity
    du = -u_0 * omega * np.sin(omega*t)
    #Acceleration
    ddu = -u_0 * omega**2 * np.cos(omega*t)

    #Plot 1.2
    if p1_2:
        print(f'K = {K:.2f} N/m \nomega = {omega:.2f} rad/s \nT = {T:.2f} s')

        plt.figure()

        plot_subplot(t, u, Time_range, 3, 1, 1, 'Displacement [m]', 'b', u_0)
        plot_subplot(t, du, Time_range, 3, 1, 2, 'Velocity [m/s]', 'r', u_0*omega)
        plot_subplot(t, ddu, Time_range, 3, 1, 3, 'Acceleration [m/s²]', 'g', u_0*omega**2)
        plt.xlabel('Time [s]')

        plt.savefig('figures/plot-1.2.pdf')

    if p1_3:
        def transform_str(str) :
            return str.replace(',', '.')
        df = pd.read_csv('Free_vibration_3Masses.csv', sep = ';')
        df = pd.read_csv('Free_vibration_3Masses.csv', sep = ';')
        df[df.columns[1:]] = df[df.columns[1:]].applymap(transform_str)
        
        f_acc = 1024 #Hz
        df.insert(0, 'Time [s]', df[df.columns[0]]/f_acc)
        column = df.columns


        h = 1/f_acc
        n = df.shape[0]
        acc = np.array(df[column[2]], dtype='float64') + 0.06846854022617188
        t = np.array(df[column[0]], dtype='float64')
        
        #i_0 = 10 * f_acc
        #i_0 = np.argmax(acc[i_0:]) + i_0
        #i_0 = np.argmax(acc)
        i_last = int(t[-1] * f_acc)
        i_0 = int(5 * f_acc)
        t = t[i_0: i_last]
        n = i_last - i_0

        print(np.mean(acc[0:1*f_acc]))


        def f(t, U):
            i = int(t/h)
            return np.array([U[1], acc[i]], dtype='float64')
        t, U = Adams_Bashforth_4(f, [0, 0], h, n, t)
        

        plt.figure()
        plt.plot(df[column[0]], np.array(df[column[2]], dtype='float64'), 'b', label = 'a_x [m/s²]')
        plt.plot(df[column[0]], np.array(df[column[3]], dtype='float64'), 'r', label = 'a_y [m/s²]')
        plt.plot(df[column[0]], np.array(df[column[4]], dtype='float64'), 'g', label = 'a_z [m/s²]')
        plt.xlabel('Time [s]')
        plt.xlim(df[column[0]][0], df[column[0]][df.shape[0]-1])
        plt.ylabel('Acceleration [m/s²]')
        plt.legend()
        plt.grid(True)
        plt.savefig('figures/plot-1.3.pdf')
        plt.figure('Velocity')
        plt.plot(t, U[:,1], 'y', label = 'v_x [m/s] (RK4)')
        plt.figure('Position')
        plt.plot(t, U[:,0], 'c', label = 'x [m] (RK4)')
        """
        y = fft(acc[i_0:i_last])
        x = fftfreq(n, h)
        plt.figure('FFT')
        plt.plot(x,y, 'b')"""
    
    
    
    
    
    plt.show()

