# command + s pour sauvegarder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Lire le fichier CSV et prendre la première feuille
df = pd.read_csv("Free_Vibration_3Masses.csv", sep=";")

# Selection première colonne
ticks = df.iloc[:, 0].array

# Selection deuxième colonne
ax = df.iloc[:, 1].array


# Selection troisième colonne
ay = df.iloc[:, 2].array

# Selection quatrième colonne
az = df.iloc[:, 3].array

# Tracer le graphe
plt.plot(ticks, ax, label="ax")
plt.plot(ticks, ay, label="ay")
plt.plot(ticks, az, label="az")
plt.xlabel('Temps [s]')
plt.ylabel('Accélération [m/s^2]')
plt.title('Vibration libre')
plt.grid()
plt.legend()
#plt.savefig("Free_Vibration_3Masses.pdf", format="pdf")
plt.show()




