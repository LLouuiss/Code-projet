import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
import math

fontsize = 10
plt.rc('font', size = fontsize)

#Data for the Newmark Method
beta = 1/6
gamma = 1/2

"""
Apply the Newmark Method to solve the equation of motion for a single degree of freedom
Input:
    t: np.array time
    u_0: np.array initial conditions
    p: np.array applied force
    m: float mass of the system
    k: float stiffness of the system
    c: float damping of the system
Output:
    U: np.array displacement, velocity and acceleration of the system
"""
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

"""
Apply the Newmark Method to solve the equation of motion for multiple degree of freedom
Input:
    t: np.array time
    u_0: np.array initial conditions of dimension Nx2
    p: np.array applied force
    K: np.array stiffness matrix of dimension NxN
    M: np.array mass matrix of dimension NxN
    C: np.array damping matrix of dimension NxN
Output:
    U: np.array displacement, velocity and acceleration of the system
"""
def Newmark_Method_M(t, u_0, P, K, M, C):
    N = len(K)
    n = len(t)
    U = np.zeros((N, 3, n))
    Q = np.zeros((N, 3, n))
    phi = phi_normalized_M(K, M)
    phi_inv = np.linalg.inv(phi)
    K_g = phi.T @ K @ phi
    C_g = phi.T @ C @ phi
    print(C_g)
    P_g = phi_inv @ P
    q_0 = np.zeros((N,2))
    for j in range(2):
        q_0[:,j] = phi_inv @ u_0[:,j] 
    if isdiag(C_g, 1e-15)==False:
        print("C_g is not diagonal")
    for i in range(N):
        Q[i] = Newmark_Method_S(t, q_0[i], P_g[i], 1, K_g[i][i], C_g[i][i])
    for j in range(3):
        U[:,j] = phi @ Q[:,j]
    return U

def normalize_mass(phi, M):
    phi_new = np.zeros(phi.shape)
    for i in range(phi.shape[1]):
        phi_new[:,i] = phi[:,i] / np.sqrt(phi[:,i].T @ M @ phi[:,i])
    return phi_new

def phi_normalized_M(K, M):
    eig_val, eig_vec = eig(K, M)
    phi = normalize_mass(eig_vec, M)
    return phi

def isdiag(A, tol):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i != j and A[i][j] > tol:
                return False
    return True

def mass_matrix(m, l):
    M = np.zeros((len(m),len(m)))
    for i in range(len(m)):
        for j in range(len(m)):
            for k in range(max(i,j), len(m)):
                M[i][j] += m[k]*l[i]*l[j]
    return M

def stiffness_matrix(m, l):
    K = np.zeros((len(m),len(m)))
    for i in range(len(m)):
        for k in range(i, len(m)):
            K[i][i] += m[k]*l[i]**2
    return K

def damping_matrix(c):
    C = np.zeros((len(c), len(c)))
    C[0][0] = c[0] + c[1]
    for i in range(1, len(c)):
        if i == len(c) - 1:
            C[i][i] = c[i]
        else : C[i][i] = c[i] + c[i+1]
        C[i-1][i] = -c[i]
        C[i][i-1] = -c[i]
    return C

def plot_initialize(title, xlabel, ylabel):
    plt.figure(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()

def plot_end(file_name):
    plt.legend()
    plt.savefig(file_name)
    plt.show()

def plot_M(t, U, title, xlabel, ylabel ):
    plot_initialize(title, xlabel, ylabel)
    for i in range(U.shape[0]):
        plt.plot(t, U[i], label="Node "+str(i))
    plt.xlim((0,20))
    plot_end(f"Assignement/figure/{title}.pdf")

if __name__=="__main__":
    print("Wrong Code!!!")