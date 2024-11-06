import numpy as np
import matplotlib.pyplot as plt

beta = 1/6
gamma = 1/2

def ground_Acceleration(t, A, T):
    A_g = A * np.ones(t.shape)
    A_g[(t/T)%2 >= 1] = -A
    A_g[t>40] = 0
    return A_g

def Newmark_Method(t, u_0, p, m, k, c):
    U = np.zeros((3, t.shape[0]))
    U[0][0] = u_0[0]
    U[1][0] = u_0[1]
    dt = t[1] - t[0]
    U[2][0] = (p[0] - c*U[1][0] - k*U[0][0])/m
    for i in range(0, t.shape[0]-1):
        du_p = U[1][i] + dt*(1 - gamma)*U[2][i]
        u_p = U[0][i] + dt*U[1][i] + dt**2*(1/2 - beta)*U[2][i]
        k_b = m/beta/dt**2 + c*gamma/beta/dt + k
        p_b = p[i + 1] + m * u_p/beta/dt**2 + c*(gamma * u_p/beta/dt - du_p)
        U[0][i + 1] = p_b/k_b
        U[2][i + 1] = (U[0][i + 1] - u_p) / (beta*dt**2)
        U[1][i + 1] = du_p + gamma*U[2][i + 1]*dt
        
    return U


if __name__ == '__main__':
    
    #Data
    E = 210e9 #Pa
    I = 7.158e-12 #m^4
    L = .259 #m
    m = .812 #kg
    g = 9.81 #m/s^2

    #Stiffness
    K = 3*E*I/(L**3)
    #Natural Cyclic Frequency
    omega = (K/m)**.5
    #Natural Period
    T = 2*np.pi/omega
    print(f'K = {K:.2f} N/m \nomega = {omega:.2f} rad/s \nT = {T:.2f} s')

    #Amplitude
    A = 0.1 * g
    #Time
    t_g = 60 #s
    u_0 = [0,0]
    C = .01*2*m*omega

    plt.figure()

    for dt in [T/100]:#T/20, T/10, T/5, T/3]:
        t = np.arange(0, t_g, dt)
        p = -m * ground_Acceleration(t, A, T)
        U = Newmark_Method(t, u_0, p, m, K, C)
        plt.plot(t, U[0], label=f'dt = {dt:.2f}s')
        #plt.plot(t, U[1], label=f'dt = {dt:.2f}')
        #plt.plot(t, U[2], label=f'dt = {dt:.2f}')
        #plt.plot(t, p, label='p')
    
    plt.xlim(0, t_g)
    plt.legend()
    plt.grid()
    plt.show()
