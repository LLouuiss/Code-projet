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


# Plot
plt.plot(ticks, ax_exp, label="Experimental acceleration")
#plt.plot(ticks, ax_theo, label="Acceleration theorique")
#plt.plot(ticks, ax_theo_adapt, label="Acceleration theorique adaptée")
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.show()

"""
plt.plot(ticks, dx, label="Deplacement experimentale")
plt.plot(ticks, dx_theo, label="Deplacement theorique")
#plt.plot(ticks, dx_theo_adapt, label="Deplacement theorique adaptée")
#plt.plot(ticks, dx_adapt, label="Deplacement theorique adaptée")
plt.legend()
plt.grid()
plt.xlabel("Temps (s)")
plt.ylabel("Deplacement (m)")
plt.show()"""











