import numpy as np
from visualize_pendulum import VisualizePendulum


# Parameter und Anfangsbedingungen
g = 9.81
L = 1
t_min = 0
t_max = 10
h = 0.01

theta_0 = np.deg2rad(45)
omega_0 = 0

# Zeitwerte und logging
t_values = np.arange(0, t_max, h)
z_values = []

# Anfangswerte
z_10 = theta_0
z_20 = omega_0

for t in t_values:    
    z_11 = z_10 + h*z_20
    z_21 = z_20 + h*(-g/L*np.sin(z_10))

    z_10 = z_11
    z_20 = z_21
    z_values.append(z_10)

viz_pendel = VisualizePendulum(t_values, z_values)
# viz_pendel.plot()
viz_pendel.animate()