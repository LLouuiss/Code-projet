import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig,solve
import math
from matplotlib.animation import FuncAnimation
from functools import partial

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
    k_b = m/beta/dt**2 + c*gamma/beta/dt + k
    U[2][0] = (p[0] - c*U[1][0] - k*U[0][0])/m
    for i in range(0, t.shape[0]-1):
        du_p = U[1][i] + dt*(1 - gamma)*U[2][i]
        u_p = U[0][i] + dt*U[1][i] + dt**2*(1/2 - beta)*U[2][i]
        p_b = p[i + 1] + m * u_p/beta/dt**2 + c*(gamma * u_p/beta/dt - du_p)
        U[0][i + 1] = p_b/k_b
        U[2][i + 1] = (U[0][i + 1] - u_p) / (beta*dt**2)
        U[1][i + 1] = du_p + gamma*U[2][i + 1]*dt
    return U

"""
This function is deprecated
"""
def Newmark_Method_M_old(t, u_0, P, K, M, C):
    N = len(K)
    n = len(t)
    U = np.zeros((N, 3, n))
    Q = np.zeros((N, 3, n))
    w, phi = phi_normalized_M(K, M)
    phi_inv = np.linalg.inv(phi)
    C_g = phi.T @ C @ phi
    P_g = phi_inv @ P
    q_0 = np.zeros((N,2))
    for j in range(2):
        q_0[:,j] = phi_inv @ u_0[:,j] 
    if isdiag(C_g, 1e-12)==False:
        print("C_g is not diagonal")
        print(C_g)
    for i in range(N):
        Q[i] = Newmark_Method_S(t, q_0[i], P_g[i], 1, w[i]**2, C_g[i][i])
    for j in range(3):
        U[:,j] = phi @ Q[:,j]
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
    U[:,0,0] = u_0[:,0]
    U[:,1,0] = u_0[:,1]
    dt = t[1] - t[0]
    K_b = M/beta/dt**2 + C*gamma/beta/dt + K
    U[:,2,0] = np.linalg.inv(M) @ (P[:,0] - C @ U[:,1,0] - K @ U[:,0,0])
    for i in range(0, t.shape[0]-1):
        dU_p = U[:,1,i] + dt*(1 - gamma)*U[:,2,i]
        U_p = U[:,0,i] + dt*U[:,1,i] + dt**2*(1/2 - beta)*U[:,2,i]
        P_b = P[:,i + 1] + M @ U_p/beta/dt**2 + C @ (gamma * U_p/beta/dt - dU_p)
        U[:,0,i + 1] = solve(K_b, P_b)
        U[:,2,i + 1] = (U[:,0,i + 1] - U_p) / (beta*dt**2)
        U[:,1,i + 1] = dU_p + gamma*U[:,2,i + 1]*dt
    return U

def normalize_mass(phi, M):
    phi_new = np.zeros(phi.shape)
    for i in range(phi.shape[1]):
        phi_new[:,i] = phi[:,i] / np.sqrt(phi[:,i].T @ M @ phi[:,i])
    return phi_new

def phi_normalized_M(K, M):
    eig_val, eig_vec = eig(K, M)
    phi = normalize_mass(eig_vec, M)
    return np.real(eig_val)**.5, phi

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
            K[i][i] += m[k]*9.81*l[i]
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

def max_lateral_displacement(U,l):
    N = U.shape[0]
    n = U.shape[1]
    x = np.zeros((N+1, n))
    for i in range(1,N+1):
        x[i] = x[i-1] + l[i-1]*np.sin(U[i-1])
    return np.max(np.abs(x))

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

def plot_mode(w, phi,l, title):
    n = len(w)
    plt.figure(title, figsize=(10,10))
    for i in range(n):
        plt.subplot(2,n//2,i+1)
        ax = plt.gca()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.grid()
        plt.ylim((-.75,0))
        plt.xlim((-.3,.3))
        plt.title(f"Mode {i+1} - w = {w[i]:.2f} rad/s")
        x = np.zeros(n+1)
        y = np.zeros(n+1)
        for j in range(1, n+1):
            x[j] = x[j-1] + l[j-1]*np.sin(phi[j-1,i])
            y[j] = y[j-1] - l[j-1]*np.cos(phi[j-1,i])
        plt.plot(x,y,"-b", label=f"Mode {i+1} - w = {w[i]:.2f} rad/s")
    plt.savefig(f"Assignement/figure/{title}.pdf")
    plt.show()


def generate_vid(fig, func, init, t, D, filename):
    print('start_animation')
    ani = FuncAnimation(fig, partial(func, U=D), frames=np.arange(0,t.size), init_func=init, interval=(t[1]-t[0])*1e-3)
    ani.save("Assignement/vid/" + filename, fps=1/(t[1]-t[0]), dpi=300)
    print("End animation")

if __name__=="__main__":
    print("Wrong Code!!!")