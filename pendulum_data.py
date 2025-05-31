from typing import Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from pendulum_ode import damped_ode 

class PendulumData:
    """
    Manages the data for the pendulum simulation, including input values,
    down-sampling, and computing a reference solution.
    """
    values_time: np.ndarray
    values_angle: np.ndarray
    values_dangle: np.ndarray
    reference: bool
    values_time_ref: Optional[np.ndarray] = None # Make Optional as it might not be computed
    values_angle_ref: Optional[np.ndarray] = None # Make Optional
    ref_step_width: float = 0.01
    init_step_width: float
    step_width: float
    omega_0_ref: float
    D_ref: float

    def __init__(self, values_time: np.ndarray, values_state: np.ndarray,
                 reference: bool = False, omega_0_ref: float = np.sqrt(9.81),
                 D_ref: float = 0.05) -> None:
        self.reference = reference
        self.omega_0_ref = omega_0_ref
        self.D_ref = D_ref
        self._assign_values(values_time, values_state)

        if self.reference:
            self._compute_and_assign_reference_solution()

    def _assign_values(self, values_time: np.ndarray, values_state: np.ndarray) -> None:
        self._assign_init_step_width(values_time)
        if self.init_step_width >= self.ref_step_width:
            self.values_time = values_time
            self.values_angle = values_state[0, :]
            self.values_dangle = values_state[1, :]
            self.step_width = self.init_step_width
        else:
            self._sample_data_down_and_assign(values_time, values_state)

    def _assign_init_step_width(self, values_time: np.ndarray) -> None:
        self.init_step_width = values_time[1] - values_time[0]

    def _sample_data_down_and_assign(self, values_time: np.ndarray, values_state: np.ndarray) -> None:
        values_time_interpolated = np.arange(values_time[0], values_time[-1] + self.ref_step_width, self.ref_step_width)
        values_angle_interpolated = np.interp(values_time_interpolated, values_time, values_state[0, :])
        values_dangle_interpolated = np.interp(values_time_interpolated, values_time, values_state[1, :])

        self.values_time = values_time_interpolated
        self.values_angle = values_angle_interpolated
        self.values_dangle = values_dangle_interpolated
        self.step_width = self.ref_step_width

    def _compute_and_assign_reference_solution(self) -> None:
        t_min: float = self.values_time[0]
        t_max: float = self.values_time[-1]
        theta_start: float = self.values_angle[0]
        omega_start: float = self.values_dangle[0]

        t_eval = np.linspace(t_min, t_max, len(self.values_time))

        sol = solve_ivp(
            damped_ode,
            (t_min, t_max),
            [theta_start, omega_start],
            method="RK45",
            t_eval=t_eval,
            args=(self.omega_0_ref, self.D_ref)
        )
        self.values_time_ref = sol.t
        self.values_angle_ref = sol.y[0]