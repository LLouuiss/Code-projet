from functions import *

#point g
#Initialize matrix
nb = 6
m = 0.2 * np.ones(nb)
l = 0.75/nb * np.ones(nb)
K = stiffness_matrix(m, l)
M = mass_matrix(m, l)

w, phi = eig(K, M)
w = np.real(w)**.5
for i in range(nb):
    print(f"natural frequency {i+1} = {w[i]:.3f} rad/s")
    print(f"natural period {i+1} = {2*math.pi/w[i]:.3f} s")
plot_mode(w, phi, l, "Mode-shape")

theta0 = np.array([np.pi/6, 0, 0, 0, 0, 0]) # theta
# Normalization by mass
phi_n = normalize_mass(phi, M)
# Initialize time-vector [s]
t_end = 20 # s
nb_steps = 2000
time = np.linspace(0,t_end,nb_steps)

q0 = phi_n.T @ M @ theta0
# Compute generalized coordinates q
# For 2 independent differential equations q'(0) = 0 and q(0) = q_0
q = np.zeros((nb,nb_steps))
for i in range(nb) :
    for t in np.arange(0,nb_steps):
        q[i,t] = q0[i] * np.cos(w[i] * time[t])

thetha = phi_n @ q
plt.plot(time, thetha[0,:], label='Theta 1')
plt.plot(time, thetha[1,:], label='Theta 2')
plt.plot(time, thetha[2,:], label='Theta 3')
plt.plot(time, thetha[3,:], label='Theta 4')
plt.plot(time, thetha[4,:], label='Theta 5')
plt.plot(time, thetha[5,:], label='Theta 6')
plt.xlim((0,20))
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid()
plt.show()

# scaling-independent coordinates
Pref = K @ theta0
Gamma = phi_n.T @ Pref
v = np.zeros((nb,nb_steps))
w2v = np.zeros((nb,nb_steps))
for i in range(nb):
    v[i,:] = q[i,:]/Gamma[i]
    w2v[i,:] = w[i]**2 * v[i,:]
    
# v theorique 
v_theo = np.zeros((nb,nb_steps))
w2v_theo = np.zeros((nb,nb_steps))
for i in range(nb):
    for t in range(nb_steps):
        v_theo[i,t] = np.cos(w[i] * time[t])/(w[i]**2)
        w2v_theo[i,t] = w[i]**2 * v_theo[i,t]
        

plt.plot(time, w2v[0,:], label='w2v 1')
plt.plot(time,w2v_theo[0,:], label='w2v theo 1')
plt.xlim((0,20))
plt.xlabel('Time [s]')
plt.ylabel('w2v')
plt.legend()
plt.grid()
plt.show()


def rayleigh_damping(w1, w2,xsi,):
    a0 = xsi*2*(w1*w2)/(w1 + w2)
    a1 =xsi* 2*(1)/(w1 + w2)
    return a0, a1

xsi = 0.01  # 1% damping for modes 1 and 2
w1, w2 = w[0], w[1]  # First two natural frequencies
# Compute Rayleigh coefficients
a0, a1 = rayleigh_damping(w1, w2, xsi)
print(f"Rayleigh coefficients: a_0 = {a0:.5f}, a_1 = {a1:.5f}")
# Compute damping matrix
C = a0 * M + a1 * K
print("Damping matrix C:")
print(C)

M_m = phi_n.T @ M @ phi_n
K_m = phi_n.T @ K @ phi_n
C_m = a0 * M_m + a1 * K_m
print("Damping matrix C_m:")
print(C_m)
C_m[np.abs(C_m) < 1e-10] = 0
print("Damping matrix C_m (with small values rounded to 0):")
print(C_m)

q_damped = np.zeros((nb, nb_steps)) 
q_dot_0 = np.zeros(nb) 
for i in range(nb):
    omega_i = w[i]
    xsi_i = C_m[i, i] / (2 * omega_i)  # Modal damping ratio
    omega_d = omega_i * np.sqrt(1 - xsi_i**2)  # Damped natural frequency
    for j in range(1, nb_steps):
        t = time[j]
        # Analytical solution for free vibration of a single DOF system
        q_damped[i, j] = (np.exp(-xsi_i * omega_i * t)* ( q0[i] * np.cos(omega_d * t)+ (q_dot_0[i] + xsi_i * omega_i * q0[i]) / omega_d * np.sin(omega_d * t)))
    
thetha_damped = phi_n @ q_damped
plt.plot(time, thetha_damped[0,:], label='Theta 1')
plt.plot(time, thetha_damped[1,:], label= 'Theta 2')
plt.plot(time, thetha_damped[2,:], label='Theta 3')
plt.plot(time, thetha_damped[3,:], label='Theta 4')
plt.plot(time, thetha_damped[4,:], label='Theta 5')
plt.plot(time, thetha_damped[5,:], label='Theta 6')
plt.xlim((0,20))
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid()
plt.show()

Pref_damped = K @ theta0
Gamma_damped = phi_n.T @ Pref
v_damped = np.zeros((nb,nb_steps))
w2v_damped = np.zeros((nb,nb_steps))
for i in range(nb):
    v_damped[i,:] = q_damped[i,:]/Gamma[i]
    w2v_damped[i,:] = w[i]**2 * v_damped[i,:]
    


# Groun dmotion 
J = np.ones(nb)
Pref_ground = M @ J
Gamma_ground = phi_n.T @ Pref_ground
