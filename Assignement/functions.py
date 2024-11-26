import numpy as np
import matplotlib.pyplot as plt

#Data for the Newmark Method
beta = 1/6
gamma = 1/2

def Newmark_Method_S(t, u_0, p, m, k, c):
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

def Newmark_Method_M(t, q0, P, K, M, C):
    Q = np.numpy((len(P), 3, len(t)))
    
    return Q

def mass_matrix(m, l):
    M = np.zeros((len(m),len(m)))
    for i in range(len(m)):
        for j in range(len(m)):
            for k in range(max(i,j), len(m)):
                M[i][j] += m[k]*l[i]*l[j]
    return M

def stiffness_matrix(m, l):
    K = np.array((len(m),len(m)))
    for i in range(len(m)):
        for k in range(i, len(m)):
            K[i][i] += m[k]*l[i]**2
    return K