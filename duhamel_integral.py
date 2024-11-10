import math
import numpy as np
import matplotlib.pyplot as plt



g = 9.81  # Acceleration due to gravity in m/s^2
A = 0.1*g  # Amplitude
Tn = 0.35  # Natural period in seconds
wn = 17.87  # Damped natural frequency in rad/s
xi = 1.39*10**(-3)  # Damping ratio (example, adjust as needed)
m = 0.812  # Mass of the structure in kg (example, adjust as needed)
wd = wn*math.sqrt(1-xi**2)
#f = omega_d / (2*math.pi)

               
tMax = 40
delT = Tn/1000
time = np.arange(0, tMax + delT, delT)


# Forcing function a(tau)
def a(tau):
    period = 2 * Tn  # Total period of the forcing function
    t_mod = tau % period  # Modulo to handle periodicity
    if t_mod < Tn:
        return A  # First half of the period
    else:
        return -A  # Second half of the period

def ground_Acceleration(t, A, T):
    A_g = A * np.ones(t.shape)
    A_g[(t/T)%2 >= 1] = -A
    A_g[t>40] = 0
    return A_g

def Du(T):
    y_A = np.exp(xi*wn*T) * m * ground_Acceleration(T, A, Tn) * np.cos(wd*T)
    y_B = np.exp(xi*wn*T) * ground_Acceleration(T, A, Tn) * np.sin(wd*T)
    AC = np.zeros(len(T))
    BC = np.zeros(len(T))
    for i in range(1, len(T)):
        AC[i] = AC[i-1] + 0.5*delT*(y_A[i] + y_A[i-1])/wd/m
        BC[i] = BC[i-1] + 0.5*delT*(y_B[i] + y_B[i-1])/wd/m
    return AC*np.exp(-xi*wn*T)*np.sin(wd*T) - BC*np.exp(-xi*wn*T)*np.cos(wd*T)

def Duhamel(T):
    U = np.zeros(len(T)) # a zero for every timestep to hold the displacements values
    
    ACum_i = 0 #initialize values for the cumulative sum used 
    BCum_i = 0 #to calculate A and B at each time step
    
    for i in range(len(T)):
        
        if i>0:
            #calculate A
            y_i = math.e**(xi*wn*T[i]) * m * a(T[i]) * math.cos(wd*T[i])
            y_im1 = math.e**(xi*wn*T[i-1]) * m * a(T[i-1]) * math.cos(wd*T[i-1])
            Area_i = 0.5*delT*(y_i + y_im1)
            ACum_i += Area_i
            A_i = (1/(m*wd))*ACum_i
            
            #calculate B
            y_i = math.e**(xi*wn*T[i]) * a(T[i]) * math.sin(wd*T[i])
            y_im1 = math.e**(xi*wn*T[i-1]) * a(T[i-1]) * math.sin(wd*T[i-1])
            Area_i = 0.5*delT*(y_i + y_im1)
            BCum_i += Area_i
            B_i = (1/(m*wd))*BCum_i
            
            #calculate the response
            U[i] = A_i*math.e**(-xi*wn*T[i])*math.sin(wd*T[i]) - B_i*math.e**(-xi*wn*T[i])*math.cos(wd*T[i])
    return U   

response = Du(time)
# Plotting the displacement response
plt.plot(time, response)
#plt.plot(time, p)
plt.title('Displacement Response of the Structure using Duhamel’s Integral')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.grid(True)
plt.show()