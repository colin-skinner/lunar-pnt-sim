import numpy as np
from dataclasses import dataclass
from typing import Callable
from scipy.linalg import block_diag
import jax.numpy as jnp
import jax
import pdb
from scipy import stats

from ..sim.simulator import rigid_body_derivative, lander_motion
from ..sim.simulator import SimParams
from ..sim.sensors import SensorSuite, SensorEnvironment
from ..sim.quaternion import unitize_state


# Generalized EKF based on Algorithms for Decision Making textbook (AA 228)
# https://algorithmsbook.com/files/dm.pdf




@dataclass
class EkfParams:
    Q: jnp.ndarray # process noise covariance
    R: jnp.ndarray # measurement noise covariance
    H: Callable[[jnp.ndarray], jnp.ndarray] # measurement Jacobian function
    h: Callable[[jnp.ndarray], jnp.ndarray] # measurement function

# @jax.jit
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
    S_inv = jnp.linalg.pinv(S) # small for weighting meas more


    n_meas = len(y)
    NIS = y @ jnp.linalg.solve(S, y) # can do this because S*(S\y) = y and we want S\y
    # NIS_per_dim = 

    # k = 3 # bound for when the sqrt kicks in (in units of meas)
    # R_scale = jnp.maximum(1.0, jnp.sqrt(NIS_per_dim) / k)
    # R_adapted = R * R_scale**2

    # k = 3 # bound for when the sqrt kicks in (in units of meas)
    # R_scale = jnp.maximum(1.0, NIS_per_dim / k**2) # now units of sigma^2
    # R_robust = R * R_scale


    # n_meas = len(y)
    # NIS = jnp.array((y @ jnp.linalg.solve(S, y)), float)
    # NIS_per_dim = NIS / n_meas  # should be ~1 for healthy filter

    # # Huber scaling: inflate R when NIS is large
    # R_scale = jnp.maximum(1.0, NIS_per_dim / 2.0)
    # R_robust = R * R_scale
    # Innovation check
    # Small y, large S --> small NIS --> very noisy
    # Large y, small S --> large NIS --> not noisy
    # NIS = y.T @ S_inv @ y


    # Chi-squared test: expected value is measurement_dim
    measurement_dim = R.shape[0]
    chi2_alpha = stats.chi2.ppf(0.99, df=measurement_dim)  # 95th percentile
    
    # Adapt R based on chi-squared consistency
    NIS_per_dim = NIS / n_meas
    R_adapted = jnp.where(
        NIS_per_dim > chi2_alpha,
        R * NIS_per_dim / chi2_alpha,
        R
    )



    # if NIS > chi2_alpha:
    #     # Innovation too large → scale up R (less trust measurements)
    #     scale_factor = NIS / chi2_expected
    #     R_adapted = R * scale_factor
    #     # trust_level = "LOW"
    # elif NIS < stats.chi2.ppf(0.05, df=measurement_dim):  # 5th percentile
    #     # Innovation too small → Q might be too large or R too small
    #     # Less common to adapt R downward; usually indicates Q tuning issue
    #     R_adapted = R
    #     # trust_level = "NOMINAL (Q check needed)"
    # else:
    #     # Innovation within expected bounds
    #     R_adapted = R
    #     # trust_level = "HEALTHY"


    # if any(np.isnan(NIS)):
    #     breakpoint()


    K = P @ H.T @ S_inv # large for weighting meas more

    x_next = x + K @ y

    q = x_next[6:10]
    q = q / jnp.linalg.norm(q)
    x_next = x_next.at[6:10].set(q)

    I = jnp.eye(len(x))
    P_next = (I - K @ H) @ P @ (I - K @ H).T + K @ R_adapted @ K.T

    return x_next, P_next

def ekf_predict_state_only(x: jnp.ndarray, a_meas: jnp.ndarray, w_meas: jnp.ndarray, sim: SimParams) -> jnp.ndarray:
    """Predict step that returns only the state (not covariance). Useful for observability Jacobians."""
    x_copy = jnp.concatenate([x[0:10], w_meas])
    force_B = a_meas * sim.body.mass_kg
    torque_B = jnp.zeros(3)
    x_next = lander_motion(x_copy, force_B, torque_B, sim.dt, sim.body.mass_kg, sim.body.I)
    return x_next

def update_sensor(name: str, freq: int, mu_pred, Sigma_pred, env, sensor_suite: SensorSuite, measurements_noisy, i):
    """Update with a sensor if its update frequency matches current timestep."""
    if freq is None or (i % freq) != 0:
        return mu_pred, Sigma_pred
    
    sensor = sensor_suite.sensors[name]
    meas = measurements_noisy[name][i]
    meas_expected = sensor.measure(mu_pred, env)
    H = sensor.jacobian(mu_pred, env)
    R = sensor.get_noise_cov(env)

    
    mu_update, Sigma_update = ekf_update(mu_pred, Sigma_pred, meas, H, meas_expected, R)
    mu_update = unitize_state(mu_update)
    if any(jnp.isnan(mu_update)):
        # print(sensor)
        print(f"{name=}")
        print(f"{mu_pred=}")
        print(f"{mu_update=}")
        print(f"{meas=}")
        print(f"{meas_expected=}")
        print(f"{H=}")
        print(f"{R=}")
        print()
        # pdb.set_trace()
    return mu_update, Sigma_update


def update_sensor_individual_NaN_check(name: str, freq: int, mu_pred, Sigma_pred, env, sensor_suite: SensorSuite, measurements_noisy, i):
    """Update with a sensor if its update frequency matches current timestep."""
    if freq is None or (i % freq) != 0:
        return mu_pred, Sigma_pred
    
    sensor = sensor_suite.sensors[name]
    meas = measurements_noisy[name][i]
    meas_expected = sensor.measure(mu_pred, env)
    H = sensor.jacobian(mu_pred, env)
    R = sensor.get_noise_cov(env)

    # Coupling?
    if name in ["laser_altimeter", "laser_velocity"]:
        H = H.at[:, 6:10].set(0.0)

    # For NaN
    invalid_mask = jnp.isnan(meas) | jnp.isnan(meas_expected)

    valid_indices = jnp.where(~invalid_mask)[0]
    if len(valid_indices) == 0:
        return mu_pred, Sigma_pred
    
    
 
    
    # Trunc
    meas_valid = meas[valid_indices]
    meas_expected_valid = meas_expected[valid_indices]
    H_valid = H[valid_indices, :]  # Only rows with valid measurements
    R_valid = R[jnp.ix_(valid_indices, valid_indices)]  # Valid submatrix


    # print(f"{H_valid=}")
    # print(f"{R_valid=}")
    # print("end")



    mu_update, Sigma_update = ekf_update(mu_pred, Sigma_pred, meas_valid, H_valid, meas_expected_valid, R_valid)
    mu_update = unitize_state(mu_update)

    if any(jnp.isnan(mu_update)):
        # print(sensor)
        print(f"{mu_pred=}")
        print(f"{meas=}")
        print(f"{meas_expected=}")
        print(f"{H=}")
        print(f"{R=}")
        print()
        print(f"{meas_valid=}")
        print(f"{meas_expected_valid=}")
        print(f"{H_valid=}")
        print(f"{R_valid=}")
        print()
        # pdb.set_trace()
    return mu_update, Sigma_update


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