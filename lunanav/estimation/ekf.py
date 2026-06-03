import numpy as np
from dataclasses import dataclass
from typing import Callable
from scipy.linalg import block_diag
import jax.numpy as jnp
import jax

from ..sim.simulator import rigid_body_derivative, lander_motion, linearized_lander_motion
from ..sim.simulator import SimParams


# Generalized EKF based on Algorithms for Decision Making textbook (AA 228)
# https://algorithmsbook.com/files/dm.pdf




@dataclass
class EkfParams:
    Q: jnp.ndarray # process noise covariance
    R: jnp.ndarray # measurement noise covariance
    H: Callable[[jnp.ndarray], jnp.ndarray] # measurement Jacobian function
    h: Callable[[jnp.ndarray], jnp.ndarray] # measurement function

@jax.jit
def ekf_predict(x: jnp.ndarray, P: jnp.ndarray, a_meas: jnp.ndarray, w_meas: jnp.ndarray,
                Q: jnp.ndarray, sim: SimParams) -> jnp.ndarray:
    
    x_copy = jnp.concatenate([x[0:10], w_meas])
    force_B = a_meas * sim.body.mass_kg
    torque_B = jnp.zeros(3) # TODO: not doing torque_B input right now
    x_next = lander_motion(x_copy, force_B, torque_B, sim.dt, sim.body.mass_kg, sim.body.I)
    
    Fd = jax.jacfwd(lambda s: lander_motion(s, force_B, torque_B, sim.dt, sim.body.mass_kg, sim.body.I))(x_copy)

    P_next = Fd @ P @ Fd.T + Q

    return x_next, P_next

@jax.jit
def ekf_update(x: jnp.ndarray, P: jnp.ndarray, meas: jnp.ndarray, H: jnp.ndarray, x_expected: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:

    y = meas - x_expected
    S = H @ P @ H.T + R

    K = P @ H.T @ jnp.linalg.pinv(S)

    x_next = x + K @ y

    q = x_next[6:10]
    q = q / jnp.linalg.norm(q)
    x_next = x_next.at[6:10].set(q)

    I = jnp.eye(len(x))
    P_next = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

    return x_next, P_next

def ekf_predict_state_only(x: jnp.ndarray, a_meas: jnp.ndarray, w_meas: jnp.ndarray, sim: SimParams) -> jnp.ndarray:
    """Predict step that returns only the state (not covariance). Useful for observability Jacobians."""
    x_copy = jnp.concatenate([x[0:10], w_meas])
    force_B = a_meas * sim.body.mass_kg
    torque_B = jnp.zeros(3)
    x_next = lander_motion(x_copy, force_B, torque_B, sim.dt, sim.body.mass_kg, sim.body.I)
    return x_next

def Qd_from_accel_white(dt, sigma_a):

    # 2x2 covariance for (pos, vel) driven by white accel noise in ONE axis
    Q1 = np.array([
        [dt**3/3, dt**2/2],
        [dt**2/2, dt     ]
    ]) * (sigma_a**2)

    # Build block-diagonal structure for x, y, z axes.
    # At this point the order is [x, xd, y, yd, z, zd]
    Q_block = block_diag(Q1, Q1, Q1)

    # Reorder rows/cols to match OUR state order: [x, y, z, xd, yd, zd]
    # This permutation swaps (pos, vel) pairs into the correct structure.
    perm = np.array([0, 2, 4,   1, 3, 5])
    Q6 = Q_block[np.ix_(perm, perm)]

    return Q6