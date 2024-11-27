from functions import *

# case 1
#Initialize matrix
m = 0.2 * np.ones(6)
l = 0.75/6 * np.ones(6)
K = stiffness_matrix(m, l)
M = mass_matrix(m, l)
C = np.zeros((6,6))
u_0 = np.zeros((6,2))

t = np.linspace(0, 20, 1000)
P = np.zeros((6, t.shape[0]))
u_0[0][0] = 30*math.pi/180
#Resolve and plot
if False:
    U = Newmark_Method_M(t, u_0, P, K, M, C)
    plot_M(t, U[:,0], "Undamped-free-vibration-Case-1", "Time [s]", "Angle [rad]")

c = 200 * 1e-4 * np.ones(6)
C = damping_matrix(c)
print(M)
print(K)
print(C)

if True:
    U = Newmark_Method_M(t, u_0, P, K, M, C)
    plot_M(t, U[:,0], "Damped-free-vibration-Case-1", "Time [s]", "Angle [rad]")
