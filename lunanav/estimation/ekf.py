import numpy as np
from dataclasses import dataclass
from typing import Callable

from ..sim.simulator import rigid_body_derivative, lander_motion, linearized_lander_motion
from ..sim.simulator import SimParams


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
    H: Callable[[np.ndarray], np.ndarray] # measurement Jacobian function
    h: Callable[[np.ndarray], np.ndarray] # measurement function

def ekf_predict(x: np.ndarray, P: np.ndarray, a_meas: np.ndarray, w_meas: np.ndarray,
                Q: np.ndarray, sim: SimParams) -> np.ndarray:
    x_copy = x.copy()
    x_copy[10:13] = w_meas
    force_B = a_meas * sim.body.mass_kg
    torque_B = np.zeros(3) # TODO: not doing torque_B input right now
    x_next = lander_motion(x_copy, force_B, torque_B, sim.dt, sim.body.mass_kg, sim.body.I)

    if any(np.isnan(x_next)):
        print(x_copy)
        print(force_B)
        print(torque_B)
        print()
        raise ValueError("EKF prediction step resulted in invalid state")
        
    Fd = linearized_lander_motion(x_copy, force_B, torque_B, sim.dt, sim.body.mass_kg, sim.body.I)

    P_next = Fd @ P @ Fd.T + Q

    # print(f"  ||Fd||={np.linalg.norm(Fd):.4f}, max={np.max(np.abs(Fd)):.4f}")
    # print(f"  Fd[6:10, 10:13] (quat-omega block):\n{Fd[6:10, 10:13]}")

    return x_next, P_next

def ekf_update(x: np.ndarray, P: np.ndarray, meas: np.ndarray, H: np.ndarray, x_expected: np.ndarray, R: np.ndarray) -> np.ndarray:

    y = meas - x_expected
    S = H @ P @ H.T + R

    K = P @ H.T @ np.linalg.pinv(S)

    x_next = x + K @ y

    q = x_next[6:10]
    q = q / np.linalg.norm(q)
    x_next = x_next.at[6:10].set(q)

    

    I = np.eye(len(x))
    P_next = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

    return x_next, P_next