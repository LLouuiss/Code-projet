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













