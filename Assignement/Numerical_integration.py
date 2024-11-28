from functions import *

# case 1
#Initialize matrix
nb = 6
m = 0.2 * np.ones(nb)
l = 0.75/nb * np.ones(nb)
K = stiffness_matrix(m, l)
M = mass_matrix(m, l)
C = np.zeros((nb,nb))
u_0 = np.zeros((nb,2))

t = np.linspace(0, 20, 24*20)
P = np.zeros((nb, t.shape[0]))
u_0[0][0] = 30*math.pi/180
#Resolve and plot
if False:
    U = Newmark_Method_M(t, u_0, P, K, M, C)
    plot_M(t, U[:,0], "Undamped-free-vibration-Case-1", "Time [s]", "Angle [rad]")

def init():
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim((-.75,.4))
    plt.ylim((-.8,0))

line = []
def update(frame, U):
    x = np.zeros(U.shape[0]+1)
    y = np.zeros(U.shape[0]+1)
    for i in range(1,x.size):
        x[i] = x[i - 1] + .75/U.shape[0]*math.sin(U[i-1][frame])
        y[i] = y[i - 1] - .75/U.shape[0]*math.cos(U[i-1][frame])
    if len(line)>0:
        l = line.pop(0)
        l.remove()
    line.append(plt.plot(x,y,"-b")[0])

if False:
    fig = plt.figure()
    generate_vid(fig, update, init, t, U[:,0], "Undamped-free-vibration-Case-1.mp4")

#Damping
c = 200 * 1e-4 * np.ones(6)
C = damping_matrix(c)
Phi__ = np.column_stack((np.hstack(K), np.hstack(M)))
alpha = solve(Phi__.T @ Phi__,Phi__.T @ np.hstack(C))
C = alpha[0]*K + alpha[1]*M

if False:
    U = Newmark_Method_M(t, u_0, P, K, M, C)
    plot_M(t, U[:,0], "Damped-free-vibration-Case-1", "Time [s]", "Angle [rad]")

if False:
    fig = plt.figure()
    generate_vid(fig, update, init, t, U[:,0], "Damped-free-vibration-Case-1.mp4")

#Case 2
nb = 3
m = 0.4 * np.ones(nb)
l = 0.75/nb * np.ones(nb)
K = stiffness_matrix(m, l)
M = mass_matrix(m, l)
C = np.zeros((nb,nb))
u_0 = np.zeros((nb,2))

t = np.linspace(0, 20, 1000)
P = np.zeros((nb, t.shape[0]))
u_0[0][0] = 30*math.pi/180
#Resolve and plot
if False:
    U = Newmark_Method_M(t, u_0, P, K, M, C)
    plot_M(t, U[:,0], "Undamped-free-vibration-Case-2", "Time [s]", "Angle [rad]")

if False:
    fig = plt.figure()
    generate_vid(fig, update, init, t, U[:,0], "Undamped-free-vibration-Case-2.mp4")
