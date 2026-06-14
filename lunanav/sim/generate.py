import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from tqdm import tqdm

from .simulator import SimResults, SimParams
from .sensors import SensorEnvironment, SensorSuite
from ..constants import GM_MOON, R_MOON, DEG_TO_RAD

@dataclass
class SatPosVel:
    r: np.ndarray
    v: np.ndarray


def make_sat_arrs(t_arr, altitude, raan, aop, inc) -> SatPosVel:
    """Generate satellite trajectory arrays for circular lunar orbit.

    Args:
        t_arr: time array [n]
        altitude: for circular orbit [m]
        raan: [deg]
        aop: arg of perigee (also anomaly for circular orbit) from raan [deg]
        inc: inclination [deg]
    """
    r_orbit = R_MOON + altitude
    v_norm = np.sqrt(GM_MOON / r_orbit)
    n_mean = np.sqrt(GM_MOON / r_orbit**3)
    n = len(t_arr)

    inc, raan, aop = np.radians([inc, raan, aop])

    r_arr = np.zeros((n, 3))
    v_arr = np.zeros((n, 3))

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(inc), -np.sin(inc)],
                   [0, np.sin(inc),  np.cos(inc)]])
    Rz = np.array([[np.cos(raan), -np.sin(raan), 0],
                   [np.sin(raan),  np.cos(raan), 0],
                   [0, 0, 1]])
    R = Rz @ Rx

    for i in range(n):
        nu = aop + n_mean * t_arr[i]
        r_orb = np.array([r_orbit * np.cos(nu), r_orbit * np.sin(nu), 0])
        v_orb = np.array([-v_norm * np.sin(nu), v_norm * np.cos(nu), 0])
        r_arr[i] = R @ r_orb
        v_arr[i] = R @ v_orb

    return SatPosVel(r_arr, v_arr)


def generate_env(results: SimResults, sim: SimParams, sats: list[SatPosVel] = None) -> list:
    """Build a SensorEnvironment for each timestep."""
    n_steps = len(results.t)
    env_arr = []

    if sats is not None:
        r_sats = jnp.array([s.r for s in sats])
        v_sats = jnp.array([s.v for s in sats])

    for i in range(n_steps):
        env = SensorEnvironment(
            t=results.t[i],
            mass=sim.body.mass_kg,
            specific_force_body=results.force_N[i]
        )
        if sats is not None:
            env.satellite_positions = r_sats[:, i, :]
            env.satellite_velocities = v_sats[:, i, :]
        env_arr.append(env)

    return env_arr


def generate_measurements(states: np.ndarray, env_arr: list[SensorEnvironment], sensor_suite: SensorSuite):
    """Generate clean and noisy measurements for all timesteps.

    Returns:
        measurements_clean: {str -> [n_steps, meas_dim]}
        measurements_noisy: {str -> [n_steps, meas_dim]}
    """
    n_steps = len(env_arr)
    measurements_clean = {}
    measurements_noisy = {}

    for sensor_name_enum, sensor in sensor_suite.sensors.items():
        measurements_clean[sensor_name_enum] = np.zeros((n_steps, sensor.meas_dim))
        measurements_noisy[sensor_name_enum] = np.zeros((n_steps, sensor.meas_dim))

    for i in tqdm(range(n_steps)):
        state = states[i]
        env = env_arr[i]

        for sensor_name_enum, sensor in sensor_suite.sensors.items():
            z_clean = sensor.measure(state, env)
            measurements_clean[sensor_name_enum][i] = z_clean

            noise = np.random.multivariate_normal(
                np.zeros(sensor.meas_dim),
                sensor.get_noise_cov(env)
            )
            measurements_noisy[sensor_name_enum][i] = np.array(z_clean) + noise

    return measurements_clean, measurements_noisy


######################################################################################################################################################
#                       For LQR Generation
######################################################################################################################################################

def aggresive_smoothing(arr: np.ndarray, indices: list):

    o = arr.copy()
    for i in indices:
        i1,i2 = i
        o[i1:i2+1] = np.linspace(arr[i1], arr[i2], i2-i1+1)
    return o


def remove_outliers(data, threshold_std=3, after_index = 0, before_index = -1):
    result = data.copy()
    
    # Detect outliers
    median = np.median(data, axis=0, keepdims=True)
    mad = np.median(np.abs(data - median), axis=0, keepdims=True)  # Median Absolute Deviation  "The influence curve and its role in robust estimation"
    outlier_mask = np.abs(data - median) > threshold_std * mad
    
    # Replace outliers with previous value (or neighbor average)
    for i in np.where(outlier_mask)[0]:
        if i < after_index:
            continue
        if i >= before_index:
            continue
        if i == 0:
            # First point: use next value
            result[i] = result[i + 1]
        else:
            # Use previous value
            result[i] = result[i - 1]
    
    return result

def get_initial_rv_state(altitiude_m: float = 20e3, downrange_angle_deg: float = 3):
    # 3 uprange in the -Y direction
    downrange_angle = downrange_angle_deg * DEG_TO_RAD
    r0_norm = R_MOON + altitiude_m # 20km altitude
    v0_norm = np.sqrt(GM_MOON / r0_norm) # circular 
    r0 = r0_norm * np.array([0, -np.sin(downrange_angle), np.cos(downrange_angle)])
    v0 = v0_norm * np.array([0, np.cos(downrange_angle), np.sin(downrange_angle)])

    s0 = np.array([
        *r0,
        *v0,
        1,0,0,0,0,0,0]) # doesn't matter for this because only r,v

    return s0,r0,v0