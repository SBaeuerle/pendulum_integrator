import numpy as np


def damped_ode(t: float, y: np.ndarray, omega_0: float, D: float):
    theta, dtheta = y
    return [dtheta, -2*D*omega_0*dtheta - omega_0**2*np.sin(theta)]