import numpy as np
from typing import Callable

def rk4_func(
    t: float, dt: float, x_prev: float, x_dot: Callable[[float, np.ndarray], np.ndarray]):

    k1 = x_dot(t, x_prev)
    k2 = x_dot(t + dt / 2, x_prev + dt * k1 / 2)
    k3 = x_dot(t + dt / 2, x_prev + dt * k2 / 2)
    k4 = x_dot(t + dt, x_prev + dt * k3)

    x_n = x_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_n

