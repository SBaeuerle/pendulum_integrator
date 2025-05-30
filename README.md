# VisualizePendulum

A simple Python package to visualize and animate a pendulum simulation.

## Description

This project provides a `VisualizePendulum` class that takes precomputed time and angle data for a simple pendulum (e.g., from an Euler integrator) and renders:

* A plot of angle vs. time with a growing trace and a moving marker.
* A side-by-side animation of the pendulum swinging, featuring a rectangular bob that rotates and a live timestamp.

Designed for ease of use in live-coding demonstrations and educational purposes.

## Features

* **Two-pane visualization**: angle–time curve and animated pendulum.
* **Customizable parameters**: pendulum length, bob size, animation speed.
* **Grid display**: gridlines for context, hidden ticks for a clean look.
* **Type-hinted, well-structured code**: ideal for review by experienced developers.

## Requirements

* Python 3.8+
* NumPy
* Matplotlib

Install via pip:

```bash
pip install numpy matplotlib
```

Or, if using Poetry:

```bash
poetry add numpy matplotlib
```

## Installation

1. Clone this repository:

   ```bash
   ```

git clone [https://github.com/yourusername/visualize-pendulum.git](https://github.com/yourusername/visualize-pendulum.git)
cd visualize-pendulum

````
2. (Optional) Create a virtual environment:
   ```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
````

3. Install dependencies:

   ```bash
   ```

pip install -r requirements.txt

````
   Or with Poetry:
   ```bash
poetry install
````

## Usage

Compute your pendulum simulation data (e.g., with an explicit Euler integrator). Then:

```python
from visualize_pendulum import VisualizePendulum

# t_values: np.ndarray of time points
# angle_deg: np.ndarray of angles in degrees

viz = VisualizePendulum(t_values, angle_deg, L=1.0, bob_size=0.1)
viz.animate(interval=20, repeat=False)
```

* `interval`: delay between frames in milliseconds (default matches simulation step).
* `repeat`: whether the animation loops (default `False`).

## Example

```python
import numpy as np
from visualize_pendulum import VisualizePendulum

# Simple Euler integration
g, L, h = 9.81, 1.0, 0.01
t_vals = np.arange(0, 10+h, h)
theta = np.deg2rad(45)
omega = 0.0
angles = [theta]
omega_vals = [omega]
for i in range(1, len(t_vals)):
    theta_new = angles[-1] + h * omega_vals[-1]
    omega_new = omega_vals[-1] - h*(g/L)*np.sin(angles[-1])
    angles.append(theta_new)
    omega_vals.append(omega_new)
angles_deg = np.rad2deg(angles)

viz = VisualizePendulum(t_vals, angles_deg)
viz.animate()
```

## Project Structure

```
visualize-pendulum/
├── visualize_pendulum.py       # Main module
├── README.md
├── requirements.txt
└── examples/
    └── example.py             # Demo script
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
