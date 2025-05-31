import numpy as np


def undamped_ode(t: float, y: np.ndarray, g: float, length_pend: float):
    theta, omega = y
    return [omega, - (g/length_pend) * np.sin(theta)]


def damped_ode(t: float, y: np.ndarray, g: float, length_pend: float, D: float):
    theta, omega = y
    omega_0 = np.sqrt(g/length_pend)
    return [omega, -2*D*omega*omega - omega_0 ^ 2 * np.sin(theta)]