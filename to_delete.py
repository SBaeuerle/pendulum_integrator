# pendulum_animation.py

import numpy as np
import matplotlib
# Optional: specify a different interactive backend if needed
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D 

# Pendulum parameters
g = 9.81
L = 1.0
h = 0.02  # time step (s)
t_max = 10.0  # total simulation time (s)

# Initial conditions
theta0 = np.deg2rad(45.0)  # initial angle in radians
omega0 = 0.0               # initial angular velocity

# Time array
t_values = np.arange(0, t_max + h, h)

# Allocate arrays for angle (theta) and angular velocity (omega)
theta_vals = np.zeros_like(t_values)
omega_vals = np.zeros_like(t_values)
theta_vals[0] = theta0
omega_vals[0] = omega0

# Perform explicit Euler integration
for i in range(1, len(t_values)):
    theta_vals[i] = theta_vals[i-1] + h * omega_vals[i-1]
    omega_vals[i] = omega_vals[i-1] + h * (-(g / L) * np.sin(theta_vals[i-1]))

# Set up the figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-L * 1.2, L * 1.2)
ax.set_ylim(-L * 1.2, L * 1.2)
ax.set_aspect('equal')
ax.grid(True)

# Initialize line and time text
line, = ax.plot([], [], lw=2, color='gray')
ax.scatter(0, 0, s=200, c='black', marker='o')

rect_length_long_side = 0.3
rect_length_short_side = 0.2
rect = Rectangle(
    (-rect_length_long_side/2, -rect_length_short_side/2),
    rect_length_long_side,
    rect_length_short_side,
    facecolor='white',
    edgecolor='black',
    linewidth=2,
    transform=ax.transData)
ax.add_patch(rect)
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    rect.set_transform(ax.transData)
    time_text.set_text('') 
    return line, rect, time_text


def update(frame):
    L_string = (L-rect_length_long_side/2)
    x = L * np.sin(theta_vals[frame])
    y = -L * np.cos(theta_vals[frame])
    
    x_line = x*L_string/L
    y_line = y*L_string/L

    line.set_data([0, x_line], [0, y_line])
    angle = np.arctan2(y, x)
    trans = Affine2D().rotate(angle).translate(x, y) + ax.transData
    rect.set_transform(trans)
    time_text.set_text(f'Time = {t_values[frame]:.2f} s')
    return line, rect, time_text

# Create animation
ani = FuncAnimation(fig, update, frames=len(t_values),
                    init_func=init, blit=True, interval=h*1000)

# Display
init()
update(0)
plt.title("Pendulum with Rotating Rectangular Bob (1st Frame)")
plt.show()


