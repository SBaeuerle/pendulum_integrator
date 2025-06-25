import numpy as np
from visualize_pendulum import VisualizePendulum

# Parameter 
g = 9.81
L = 0.5
m = 0.2
d =  0.2

# Zeit
t_end = 5
h = 0.1

# Speicherplatz schonmal belegen
t_values = np.arange(0, t_end, h)
number_of_steps = len(t_values)-1
x = np.zeros_like(t_values)
v = np.zeros_like(t_values)

# Anfangebdingungen
x[0] = np.deg2rad(75)
v[0] = 0

# Euler explizit
for j in range(0,number_of_steps):
    x[j+1] = x[j] + h*v[j]
    v[j+1] = v[j] + h*(-d/m*v[j]-g/L*np.sin(x[j]))


show_reference = True
results = {
    'Euler_explicit': (t_values, np.array([x,v])),
}

viz_pendel = VisualizePendulum(results, show_reference, m,d,g,L)
viz_pendel.animate()