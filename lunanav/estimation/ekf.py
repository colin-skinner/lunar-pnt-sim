import numpy as np
from dataclasses import dataclass
from typing import Callable

from ..sim.math.rigid_body import rigid_body_derivative, linearized_rigid_body_derivative
from ..sim.math.integration import rk4_func
from ..sim.simulator import SimParams, propagate


# Generalized EKF based on Algorithms for Decision Making textbook (AA 228)
# https://algorithmsbook.com/files/dm.pdf


def meas_gyro(state):
    return state[10:13]

def gyro_jacobian(state):
    del state
    J = np.zeros((3,13))
    J[:,10:13] = np.eye(3)
    return J





@dataclass
class EkfParams:
    Q: np.ndarray # process noise covariance
    R: np.ndarray # measurement noise covariance

def ekf_predict(x: np.ndarray, P: np.ndarray, a_meas: np.ndarray, w_meas: np.ndarray,
                Q: np.ndarray, sim: SimParams) -> np.ndarray:
    x[10:13] = w_meas
    force = a_meas * sim.body.mass_kg
    torque = np.zeros(3) # TODO: add torque measurement
    x_next = propagate(x, force, torque, sim, mu = 0)

    if any(np.isnan(x_next)):
        print(x)
        print(force)
        print(torque)
        print()
        raise ValueError("EKF prediction step resulted in invalid state")
    
    F = linearized_rigid_body_derivative(x, force, torque, sim.body.mass_kg, sim.body.I)
    Fd = np.eye(13) + F * sim.dt

    P_next = Fd @ P @ Fd.T + Q

    return x_next, P_next

# def ekf_update(x: np.ndarray, P: np.ndarray, z: np.ndarray, params: EkfParams, accel: np.ndarray) -> np.ndarray:
#     H = params.H(x, accel)

#     y = z - params.h(x, accel)
#     S = H @ P @ H.T + params.R

#     K = P @ H.T @ np.linalg.pinv(S)

#     x_next = x + K @ y

#     I = np.eye(len(x))
#     P_next = (I - K @ H) @ P @ (I - K @ H).T + K @ params.R @ K.T

#     return x_next, P_next