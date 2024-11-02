import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def Adams_Bashforth_4(f, U_0, h, n, t) :
    U = np.zeros((n, len(U_0)), dtype='float64')
    U[0] = U_0
    U[1] = U[0] + h/2*(f(t[0], U[0]) + f(t[1], U[1]))
    U[2] = U[1] + h/12*(-f(t[0],U[0]) + 8*f(t[1], U[1]) + 5*f(t[2], U[2]))
    for i in range(3, n):
        U[i] = U[i-1] + h/24*(f(t[i-3], U[i-3]) - 5*f(t[i-2], U[i-2]) + 19*f(t[i-1], U[i-1]) + 9*f(t[i], U[i]))
    return t, U

def bisect(f, a, b, tol, max_iter):
    f_a = f(a)
    f_b = f(b)
    if f_a*f_b > 0:
        return None
    for i in range(max_iter):
        x = (a + b)/2
        f_a = f(a)
        f_x = f(x)
        if f_x == 0 or (b - a)/2 < tol:
            return x
        if f_x*f_a < 0:
            b = x
        else:
            a = x
    return x

if __name__ == '__main__':
    def transform_str(str) :
        return str.replace(',', '.')
    df = pd.read_csv('data/Measure_241023_154019.csv', sep = ';')
    df[df.columns[1:]] = df[df.columns[1:]].applymap(transform_str)
        
    f_acc = 1024 #Hz
    df.insert(0, 'Time [s]', df[df.columns[0]]/f_acc)
    column = df.columns

    h = 1/f_acc
    n = df.shape[0]
    acc = np.array(df[column[2]], dtype='float64')
    t = np.array(df[column[0]], dtype='float64')

    i_0 = 0 * f_acc
    #i_0 = np.argmax(acc[i_0:]) + i_0
    i_last = int(20 * f_acc) #29.952231

    t = t[i_0: i_last]
    n = i_last - i_0

    print(np.mean(acc[0:int(1.6*f_acc)]))

    
    def nf(x):
        def f1(t, U):
            i = int(t/h)
            return np.array([U[1], acc[i]+x], dtype='float64')
        return Adams_Bashforth_4(f1, [ 0, 0], h, n, t)[1][-1][0]
    x = bisect(nf, 0, .4, 1e-6, 100)
    print(x)

    acc += x
    def f(t, U):
        i = int(t/h)
        return np.array([U[1], acc[i]], dtype='float64')
    t, U = Adams_Bashforth_4(f, [0, 0], h, n, t)

    if True:
        plt.figure()
        plt.plot(df[column[0]], np.array(df[column[2]], dtype='float64'), 'b', label = 'a_x [m/s²]')
        plt.xlabel('Time [s]')
        plt.xlim(df[column[0]][0], df[column[0]][df.shape[0]-1])
        plt.ylabel('Acceleration [m/s²]')
        plt.legend()
        plt.grid(True)
        plt.figure('Velocity')
        plt.plot(t, U[:,1], 'y', label = 'v_x [m/s]')
        plt.figure('Position')
        plt.plot(t, U[:,0], 'c', label = 'x [m]')
        plt.show()