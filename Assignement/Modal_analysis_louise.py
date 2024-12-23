#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:42:20 2024

@author: louiseverschuere
"""
import numpy as np
import matplotlib.pyplot as plt
from functions import *

#point g
#Initialize matrix
nb = 6
m = 0.2 * np.ones(nb)
l = 0.75/nb * np.ones(nb)
K = stiffness_matrix(m, l)
M = mass_matrix(m, l)


#### QUESTION 3.1 ######
w, phi = eig(K, M)
w = np.real(w)**.5
# Trier les fréquences naturelles w et réorganiser les modes propres phi
indices = np.argsort(w)  # Obtenir les indices pour trier w par ordre croissant
w = w[indices]           # Trier w
phi = phi[:, indices]    # Réorganiser les colonnes de phi selon les indices

# Affichage des fréquences naturelles triées
for i in range(nb):
    print(f"natural frequency {i+1} = {w[i]:.3f} rad/s")
    print(f"natural period {i+1} = {2*math.pi/w[i]:.3f} s")

# Tracer les modes propres triés
plot_mode(w, phi, l, "Mode-shape (trié)")



#### QUESTION 3.2 ######

phi_n = normalize_mass(phi, M)
U_0 = [np.pi/6, 0,0,0,0,0] #projetter un déplacement sur chacun des modes propres ?
q_0 = phi_n.T @ M @ U_0 #Cette projection indique à quel point chaque mode propre est activé par le déplacement initial.

# q_0[j] indique que le mode i contribue fortement au déplacement ititial ?
#mode 5 est en réalité le mode 1 car w le plus petit, contribue le plus au déplacement
#sa forme propre contribue le plus au déplacement

# Tracé des contributions modales
plt.bar(range(1, nb + 1), np.abs(q_0))
plt.xlabel('$\phi$')
plt.ylabel('Initial modal contribution')
#plt.title('Contributions modales à la rotation initiale')
plt.show()


#### QUESTION 3.3 ######

# Initialize time-vector [s]
t_end = 20
nb_steps = 2000
time = np.linspace(0,t_end,nb_steps)

# Compute generalized coordinates q_t
# For 2 independent differential equations q'(0) = 0 and q(0) = q_0
q_t = np.zeros((nb,nb_steps))
for i in range(nb) :
    for t in np.arange(0,nb_steps):
        q_t[i,t] = q_0[i] * np.cos(np.real(w[i]) * time[t])        

# Plot the modal contributions as functions of time
for i in range(nb) :
    plt.plot(time, q_t[i], linewidth=0.5)
plt.xlabel('Time $t$ $[s]$')
plt.ylabel('Modal contribution')
plt.grid(True)
plt.legend(['Mode 1', 'Mode 2','Mode 3', 'Mode 4','Mode 5', 'Mode 6'])
plt.savefig(f"/Users/louiseverschuere/Documents/LGCIV2042_DoS/Assignement/Modal_contrib3b.pdf")
plt.show()


# Compute omega_j^2 * q_j(t) for each mode
omega_squared_q_t = np.zeros_like(q_t)
for j in range(nb):
    omega_squared_q_t[j, :] = w[j]**2 * q_t[j, :]
# Plot omega_j^2 * q_j(t) for each mode
for i in range(nb):
    plt.plot(time, omega_squared_q_t[i], linewidth=0.5)
plt.xlabel('Time $t$ $[s]$')
plt.ylabel('$\omega_j^2 q_j(t)$ ')
plt.grid(True)
plt.legend([f'Mode {i+1}' for i in range(nb)])  # Légende dynamique pour chaque mode
plt.savefig(f"/Users/louiseverschuere/Documents/LGCIV2042_DoS/Assignement/w2qj_3c.pdf")
plt.show()

#U(t) pour chaque noeud
U_t = np.zeros((nb,nb_steps))
for i in range(nb):
    for j in range(nb):
        for t in np.arange(0, nb_steps):
            U_t[i,t] += phi_n[i,j] * q_t[j,t]
# Plot the displacement of each node
for i in range(nb):
    plt.plot(time, U_t[i,:], label=f'DoF {i+1}', linewidth=0.5)
#plt.plot(time, (3*U_t[0,:]-6*U_t[1,:])/(7*L), linewidth=0.5) #condensed DoF calculation
plt.xlabel('Time $[s]$')
plt.ylabel('Displacement [rad]')
plt.grid(True)
plt.legend()
plt.savefig(f"/Users/louiseverschuere/Documents/LGCIV2042_DoS/Assignement/Ut_test.pdf")
plt.show()


#### QUESTION 3.4 ######
def rayleigh_damping(w1, w2,xsi,):
    #Calculate Rayleigh damping coefficients a0 and a1.

    # Solve for alpha and beta
    a0 = xsi*2*(w1*w2)/(w1 + w2)
    a1 =xsi* 2*(1)/(w1 + w2)
    return a0, a1

# Input: natural frequencies and damping ratios
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
#C_m2 = phi_n.T @ C @ phi_n donne le meme résultat que au dessus
C_m[np.abs(C_m) < 1e-10] = 0
print("Damping matrix C_m (with small values rounded to 0):")
print(C_m)

q_dot_0 = np.zeros(nb)
# Solve modal equations of motion
q_t2 = np.zeros((nb, nb_steps))  # Modal displacements
q_dot_t = np.zeros((nb, nb_steps))  # Modal velocities
q_t2[:, 0] = q_0
q_dot_t[:, 0] = q_dot_0

for i in range(nb):  # Solve for each mode
    omega_i = w[i]
    xsi_i = C_m[i, i] / (2 * omega_i)  # Modal damping ratio
    omega_d = omega_i * np.sqrt(1 - xsi_i**2)  # Damped natural frequency

    for j in range(1, nb_steps):
        t = time[j]

        # Analytical solution for free vibration of a single DOF system
        q_t2[i, j] = (np.exp(-xsi_i * omega_i * t)
                  * ( q_0[i] * np.cos(omega_d * t)
                      + (q_dot_0[i] + xsi_i * omega_i * q_0[i]) / omega_d * np.sin(omega_d * t)
                    )
                    )
# Compute the total response in physical coordinates
U_t = np.zeros((nb, nb_steps))  # Displacements in physical coordinates
for i in range(nb):
    for j in range(nb):
        U_t[i, :] += phi_n[i, j] * q_t2[j, :]
# Compute omega_j^2 * q_j(t) for each mode
omega_squared_q_t2 = np.zeros_like(q_t2)
for j in range(nb):
    omega_squared_q_t2[j, :] = w[j]**2 * q_t2[j, :]
# Plot omega_j^2 * q_j(t) for each mode
for i in range(nb):
    plt.plot(time, omega_squared_q_t2[i], linewidth=0.5)
plt.xlabel('Time $t$ $[s]$')
plt.ylabel('$\omega_j^2 q_j(t)$ ')
plt.grid(True)
plt.legend([f'Mode {i + 1}' for i in range(nb)])
plt.savefig(f"/Users/louiseverschuere/Documents/LGCIV2042_DoS/Assignement/w2qj_3d.pdf")
plt.show()
# Plot the displacement of each node
for i in range(nb):
    plt.plot(time, U_t[i, :], label=f'DoF {i + 1}', linewidth=0.5)
plt.xlabel('Time $t$ $[s]$')
plt.ylabel('Displacement $\\theta(t)$ [rad]')
plt.grid(True)
plt.legend()
plt.savefig(f"/Users/louiseverschuere/Documents/LGCIV2042_DoS/Assignement/displ_3d.pdf")
plt.show()



#### QUESTION 3.5 ######

# Initialize time-vector [s]
t_max = 30
steps = 2000
temps = np.linspace(0,t_max,steps)

#définitions
T1 = 0.7 #T_bar= [0.7,1.4,2]
w1 = 2*np.pi/T1 #w_bar = [2 * np.pi / Ti for Ti in T_bar]
g = 9.81
J = np.ones(nb) # Calcul de J (vecteur spatial), Vecteur unitaire pour un chargement uniforme

# Matrice de transformation B
def B(l,nb):
    B = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(i, nb):
            B[i, j] = l[i]
    return B

#P(t) = P_ref * lambda(t)
def lambda_t(t) :
    return -1* 0.1 * g* np.sin(w1 * t)
P_ref = B(l,nb).T @ M @ J
Gamma = np.zeros(nb)  
P_ref_modal = np.zeros(nb)  
for j in range(nb):
    M_j = phi_n[:, j].T @ M @ phi_n[:, j]
    Gamma[j] = (phi_n[:, j].T @ M @ J) / M_j
    P_ref_modal[j] = Gamma[j] * M_j 
print("Contributions modales (P_ref_modal):", P_ref_modal)

# Solving differential equations for modal coordinates with time-dependent force
q_t3 = np.zeros((nb, steps))  # Store the modal responses over time

for i in range(nb):  # Iterate over each mode
    k = K_m[i][i]  # Modal stiffness
    w_i = np.real(w[i])  # Natural frequency of the mode
    xsi = C_m[i][i] / (2 * w_i)  # Damping ratio for the mode
    ratio_w = w1 / w_i  # Ratio between forcing frequency and modal frequency
    w_d = w_i * np.sqrt(1 - xsi**2)  # Damped natural frequency
    p_ref = P_ref_modal[i]
    # Initialize transient response coefficients
    A, B = 0, 0
    for j in range(steps):
        t = temps[j]

        # Compute the modal force as a function of time
        P_m_t = p_ref* lambda_t(t)  

        # Compute steady-state response coefficients
        C = (P_m_t / k) * (1 - ratio_w**2) / (((1 - ratio_w**2)**2) + (2 * xsi * ratio_w)**2)
        D = (P_m_t / k) * (-2 * xsi * ratio_w) / (((1 - ratio_w**2)**2) + (2 * xsi * ratio_w)**2)

        # Transient response coefficients (update dynamically based on time-dependent force)
        A = -D
        B = A * xsi * w_i / w_d - C * w1 / w_d

        # Compute the modal displacement at time t
        transient = np.e**(-xsi * w_i * t) * (A * np.cos(w_d * t) + B * np.sin(w_d * t))
        steady_state = D * np.cos(w1 * t) + C * np.sin(w1 * t)
        q_t3[i, j] = transient + steady_state

# Plot the modal contributions as functions of time
for i in range(nb):  # Plot for each mode
    plt.plot(temps, q_t3[i], linewidth=0.5, label=f'Mode {i + 1}')
plt.xlabel('Time $t$ $[s]$')
plt.ylabel('Modal Contribution $q_j(t)$')
plt.title('Modal Contributions to Displacements')
plt.grid(True)
plt.legend()
plt.savefig(f"/Users/louiseverschuere/Documents/LGCIV2042_DoS/Assignement/modal_contrib3e.pdf")
plt.show()

# Calculate the displacement of each node as a function of Time

U_t2 = np.zeros((nb,steps))

for i in range(nb):
    for j in range(nb):
        for t in np.arange(0, steps):
            U_t2[i,t] += phi_n[i,j] * q_t3[j,t]
# Plot the displacement of each node

for i in range(nb):
    plt.plot(temps, U_t2[i,:], linewidth=0.5)
#plt.plot(time, (3*U_t[0,:]-6*U_t[1,:])/(7*L), linewidth=0.5)
plt.xlabel('Time $t$ $[s]$')
plt.ylabel('Displacement [mm]')
plt.grid(True)
plt.savefig(f"/Users/louiseverschuere/Documents/LGCIV2042_DoS/Assignement/displ_3e.pdf")
plt.legend(['DoF 1 [rad]', 'DoF 2 [rad]', 'DoF 3 [rad]','DoF 4 [rad]', 'DoF 5 [rad]', 'DoF 6 [rad]'])

















