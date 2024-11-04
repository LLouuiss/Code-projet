# command + s pour sauvegarder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Lire le fichier CSV et prendre la première feuille
df = pd.read_csv("Free_vibration_3Masses.csv", sep=";")

# Selection première colonne
ticks = df.iloc[:, 0]
ticks = ticks.astype(float)
Fs= 1024 # Hz freq d'echantillonage
ticks = ticks/Fs # sec temps en seconde

# Selection acceleration experimentale
ax_exp = df.iloc[:, 1].array
ax_exp= [x.replace(',', '.') for x in ax_exp]
ax_exp=[float(x) for x in ax_exp]


# Calcule deplacement experimentale
omega = 17.87 # rad/s
omega_adapt = 2*np.pi/0.4 # rad/s
print(omega_adapt)
dx = [-x/(omega**2) for x in ax_exp]
dx_adapt = [-x/(omega_adapt**2) for x in ax_exp]

# Acceleration théorique
u_0 = 0.05 # m
ax_theo = -(omega**2) * u_0 * np.cos(omega*ticks)
ax_theo_adapt = -(omega_adapt**2) * u_0 * np.cos(omega_adapt*ticks)
# Deplacement théorique
dx_theo = u_0 * np.cos(omega*ticks)
dx_theo_adapt = u_0 * np.cos(omega_adapt*ticks)


"""
# Subplot Acceleration theo vs exp (2 graphiques different)

plt.subplot(2, 1, 1)
plt.plot(ticks, ax_exp, label="Experimental acceleration")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(ticks, ax_theo, label="Theoretical acceleration")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.savefig("figures/VS_ACC.pdf")
plt.show()

# Subplot Deplacement theo vs exp (2 graphiques different)
plt.subplot(2, 1, 1)
plt.plot(ticks, dx, label="Experimental displacement")
plt.ylabel("Displacement (m)")
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(ticks, dx_theo, label="Theoretical displacement")
plt.ylabel("Displacement (m)")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.savefig("figures/VS_DEPL.pdf")
plt.show()

# Subplot Acceleration exp vs the adaptée (2 graphiques different)
plt.subplot(3, 1, 1)
plt.plot(ticks, ax_theo_adapt, label="Adapted theoretical acceleration")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(ticks, dx_adapt, label="Adapted experimental displacement")
plt.ylabel("Displacement (m)")
plt.legend()
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(ticks, dx_theo_adapt, label="Adapted theoretical displacement")
plt.ylabel("Displacement (m)")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.savefig("figures/VS_ACC_DEPL_ADAPT.pdf")
plt.show()











# Acceleration
plt.plot(ticks, ax_exp, label="Experimental acceleration")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.savefig("figures/plot_acc_exp.pdf")
plt.show()

plt.plot(ticks, ax_theo, label="Theoretical acceleration")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.savefig("figures/plot_acc_the.pdf")
plt.show()

plt.plot(ticks, ax_theo_adapt, label="Adapted theoretical acceleration")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.savefig("figures/plot_acc_the_adapt.pdf")
plt.show()

# Deplacement
plt.plot(ticks, dx, label="Experimental displacement")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.savefig("figures/plot_depl_exp.pdf")
plt.show()

plt.plot(ticks, dx_theo, label="Theoretical displacement")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.savefig("figures/plot_depl_the.pdf")
plt.show()

plt.plot(ticks, dx_theo_adapt, label="Adapted theoretical displacement")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.savefig("figures/plot_depl_the_adapt.pdf")
plt.show()

plt.plot(ticks, dx_adapt, label="Adapted experimental displacement")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.savefig("figures/plot_depl_exp_adapt.pdf")
plt.show()

# Plot acceleration et deplacement
plt.plot(ticks, ax_exp, label="Experimental acceleration")
plt.plot(ticks, ax_theo, label="Theoretical acceleration")
plt.plot(ticks, ax_theo_adapt, label="Adapted theoretical acceleration")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.savefig("figures/plot_acc.pdf")
plt.show()

plt.plot(ticks, dx, label="Experimental displacement")
plt.plot(ticks, dx_theo, label="Theoretical displacement")
plt.plot(ticks, dx_theo_adapt, label="Adapted theoretical displacement")
plt.plot(ticks, dx_adapt, label="Adapted experimental displacement")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.savefig("figures/plot_depl.pdf")
plt.show()
"""





# Plot acceleration, vitesse et deplacement pour Equation of motion of the damped system with its initial conditions
#Data
E = 210e9 #Pa
I = 7.158e-12 #m^4
L = .259 #m
m = .812 #kg
u_0 = .05 # m
du_0 = 0
#Stiffness
K = 3*E*I/(L**3)
#Natural Cyclic Frequency
omega = (K/m)**.5
omega = omega_adapt
#Natural Period
T = 2*np.pi/omega
Time_range = (0, 10)
nb = 100
t = np.arange(Time_range[0], Time_range[1] + T/nb, T/nb)
# Damping ratio
xi = 1.39 * 10**-3
# Damped natural frequency
omega_d = omega * (1 - xi**2)**.5
print(f'K = {K:.2f} N/m \nomega = {omega:.2f} rad/s \nT = {T:.2f} s \nxi = {xi:.2f} \nomega_d = {omega_d:.2f} rad/s')


u_damped = np.exp(-xi*omega*t) * (u_0 * np.cos(omega_d*t) + (du_0 + xi*omega*u_0) * np.sin(omega_d*t) / omega_d)
du_damped = -xi * np.exp(-xi*omega*t) * (omega*u_0 * np.cos(omega_d*t) + (du_0 + xi*omega*u_0) * np.sin(omega_d*t)/omega_d) + np.exp(-xi*omega*t) * (-u_0 * omega_d * np.sin(omega_d*t) + (du_0 + xi*omega*u_0) * omega_d * np.cos(omega_d*t)/omega_d)
ddu_damped = -xi**2 * omega**2 * np.exp(-xi*omega*t) * (u_0 * np.cos(omega_d*t) + (du_0 + xi*omega*u_0) * np.sin(omega_d*t)/omega_d) - 2 * xi * omega * np.exp(-xi*omega*t) * (-u_0 * omega_d * np.sin(omega_d*t) + (du_0 + xi*omega*u_0) * omega_d * np.cos(omega_d*t)/omega_d) + np.exp(-xi*omega*t) * (-u_0 * omega_d**2 * np.cos(omega_d*t) - (du_0 + xi*omega*u_0) * omega_d**2 * np.sin(omega_d*t)/omega_d)

# Print equations of motion
print(f'u(t) = {u_0} * exp(-{xi}*{omega}*t) * (cos({omega_d}*t) + ({du_0} + {xi}*{omega}*{u_0}) * sin({omega_d}*t) / {omega_d})')
print(f'du(t) = -{xi} * exp(-{xi}*{omega}*t) * ({omega}*{u_0} * cos({omega_d}*t) + ({du_0} + {xi}*{omega}*{u_0}) * sin({omega_d}*t)/{omega_d}) + exp(-{xi}*{omega}*t) * (-{u_0} * {omega_d} * sin({omega_d}*t) + ({du_0} + {xi}*{omega}*{u_0}) * {omega_d} * cos({omega_d}*t)/{omega_d})')
print(f'ddu(t) = -{xi}**2 * {omega}**2 * exp(-{xi}*{omega}*t) * ({u_0} * cos({omega_d}*t) + ({du_0} + {xi}*{omega}*{u_0}) * sin({omega_d}*t)/{omega_d}) - 2 * {xi} * {omega} * exp(-{xi}*{omega}*t) * (-{u_0} * {omega_d} * sin({omega_d}*t) + ({du_0} + {xi}*{omega}*{u_0}) * {omega_d} * cos({omega_d}*t)/{omega_d}) + exp(-{xi}*{omega}*t) * (-{u_0} * {omega_d}**2 * cos({omega_d}*t) - ({du_0} + {xi}*{omega}*{u_0}) * {omega_d}**2 * sin({omega_d}*t)/{omega_d})')

# Subplot
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(t, u_damped, label="Displacement [m]", color="blue")
plt.ylabel("Displacement [m]")
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(t, du_damped, label="Velocity [m/s]", color="red")
plt.ylabel("Velocity [m/s]")
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(t, ddu_damped, label="Acceleration [m/s²]", color="green")
plt.ylabel("Acceleration [m/s²]")
plt.grid()
plt.xlabel("Time [s]")
plt.savefig("figures/plot_damped.pdf")
plt.show()

# Acceleration
plt.plot(ticks, ax_exp, label="Experimental acceleration")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.savefig("figures/plot_acc_exp.pdf")
plt.show()







