import numpy as np
from visualize_pendulum import VisualizePendulum

# Parameter und Anfangsbedingungen
g = 9.81
L = 1
omega_0 = np.sqrt(g/L)
D = 0.05
t_min = 0
t_max = 10
h = 0.01

theta_0 = np.deg2rad(45)
dtheta_0 = 0

# Zeitwerte und logging
t_values = np.arange(0, t_max, h)

# Speicherplatz alloquieren
u = np.zeros((2, len(t_values)))

# Anfangswerte
u[0, 0] = theta_0
u[1, 0] = dtheta_0

for n in range(0, len(t_values)-1):    
    u[0, n+1] = u[0, n] + h*u[1, n]
    u[1, n+1] = u[1, n] + h*(-2*D*omega_0*u[1, n] - omega_0**2*np.sin(u[0, n]))

show_reference = True
viz_pendel = VisualizePendulum(t_values, u, show_reference, omega_0, D)
viz_pendel.animate()
# viz_pendel.plot()