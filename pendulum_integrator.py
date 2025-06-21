import numpy as np
from visualize_pendulum import VisualizePendulum

# Parameter 
g = 9.81
L = 0.5
m = 0.2
d =  0.2

# Anfangsbedingungen
t_end = 5
h = 0.01


# Anfangsbedingungen
theta_0 = np.deg2rad(45)
dtheta_0 = 0 

# Speicherplatz schonmal belegen
t_values = np.arange(0, t_end, h)
u = np.zeros((2, len(t_values)))


u[0,0] = theta_0
u[1,0] = dtheta_0

for n in range(0, len(t_values)-1):
    u[0, n+1] = u[0,n] + h*u[1,n]
    u[1, n+1] = u[1,n] + h*(-d/m*u[1,n]-g/L*np.sin(u[0,n]))                            


show_reference = True
results = {
    'Euler_explicit': (t_values, u),
}

viz_pendel = VisualizePendulum(results, show_reference, m,d,g,L)
viz_pendel.animate()
