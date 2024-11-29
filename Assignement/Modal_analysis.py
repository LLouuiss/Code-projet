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
