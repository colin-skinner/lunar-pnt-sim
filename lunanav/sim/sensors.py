import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from enum import Enum

from jax.numpy.linalg import norm
from .quaternion import quat_apply, angle_axis_to_q, hamilton_product, unit, conj
from ..constants import R_MOON


class SensorName(Enum):
    """Enumeration of sensor types for type-safe references."""
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    LASER_ALTIMETER = "laser_altimeter"
    LASER_VELOCITY = "laser_velocity"
    STAR_TRACKER = "star_tracker"
    DOPPLER = "doppler"
    RANGE_TRACKER = "range_tracker"
    TERRAIN_RELATIVE_NAV = "terrain_relative_nav"


# @dataclass
# class SensorNoises:
#     accel: np.ndarray = field(default_factory=lambda: np.zeros((3,3)))
#     gyro: np.ndarray = field(default_factory=lambda: np.zeros((3,3)))
#     laser_alt: np.ndarray = field(default_factory=lambda: np.zeros((4,4)))
#     laser_vel: np.ndarray = field(default_factory=lambda: np.zeros((4,4)))
#     star_tracker: np.ndarray = field(default_factory=lambda: np.zeros((4,4)))
#     range_tracker: np.ndarray = field(default_factory=lambda: np.zeros((6,6)))

# ####################################################################################################
# #                                       Accel
# ####################################################################################################

# def meas_accel(accel_body: np.ndarray, R: np.ndarray, orientation: np.ndarray = None) -> np.ndarray:
#     """Measure acceleration with additive Gaussian noise."""
#     del orientation  # TODO
#     if R.shape != (3, 3):
#         raise ValueError(f"Expected R shape (3,3), got {R.shape}")
#     return accel_body + np.random.multivariate_normal(np.zeros(3), R)

# ####################################################################################################
# #                                       Gyro
# ####################################################################################################

# def meas_gyro(gyro_body: np.ndarray, R: np.ndarray, orientation: np.ndarray = None) -> np.ndarray:
#     """Measure angular velocity with additive Gaussian noise."""
#     del orientation  # TODO
#     if R.shape != (3, 3):
#         raise ValueError(f"Expected R shape (3,3), got {R.shape}")
#     return gyro_body + np.random.multivariate_normal(np.zeros(3), R)

# ####################################################################################################
# #                                       Line-of-sight Distance
# ####################################################################################################
ANGLE = 25
los_vectors = jnp.array([
    quat_apply(angle_axis_to_q(ANGLE, [-1,1,0], degrees=True), [0,0,-1]),
    quat_apply(angle_axis_to_q(ANGLE, [-1,-1,0], degrees=True), [0,0,-1]),
    quat_apply(angle_axis_to_q(ANGLE, [1,-1,0], degrees=True), [0,0,-1]),
    quat_apply(angle_axis_to_q(ANGLE, [1,1,0], degrees=True), [0,0,-1])
])

def get_los_vectors():
    """M: sensor frame"""
    return los_vectors # already calculated when module is imported, so only calculated once

@jax.jit
def dist_from_los(state: jnp.ndarray) -> jnp.ndarray:
    """Compute distance to lunar surface from line-of-sight vectors."""
    # TODO: deal with tilting past 90º
    r, q_B2L = state[0:3], state[6:10]
    vecs_body = get_los_vectors()
    distances = []

    for v in vecs_body:
        vec_inertial = quat_apply(q_B2L, v)
        tilt_rad = jnp.arccos(jnp.dot(r, vec_inertial) / norm(r) / norm(vec_inertial))
        alt = norm(r) - R_MOON
        dist = alt / jnp.cos(tilt_rad)
        distances.append(jnp.abs(dist))

    return jnp.array(distances)

# # """Could be a better one"""
# # def dist_from_los(state):
# #     """
# #     Find intersection of LOS rays with lunar sphere.
# #     LOS ray: r + t * d, where d is LOS direction in inertial frame.
# #     Sphere: |x| = R_MOON.
    
# #     Solve: |r + t*d|^2 = R_MOON^2
# #     Quadratic in t: t^2 + 2*(r·d)*t + |r|^2 - R_MOON^2 = 0
# #     """
# #     r, q_B2L = state[0:3], state[6:10]
# #     vecs_body = get_los_vectors()
# #     distances = []

# #     for v in vecs_body:
# #         d = quat_apply(q_B2L, v)
# #         d = d / jnp.linalg.norm(d)  # ensure unit vector
        
# #         # Quadratic coefficients
# #         b = jnp.dot(r, d)
# #         c = jnp.dot(r, r) - R_MOON**2
        
# #         discriminant = b**2 - c
        
# #         # Two solutions: t = -b ± sqrt(discriminant)
# #         # Pick the smaller positive one (closer intersection)
# #         t = -b - jnp.sqrt(jnp.maximum(discriminant, 0))
        
# #         # If discriminant < 0, no intersection (LOS misses surface)
# #         dist = jnp.where(discriminant > 0, t, jnp.nan)
# #         # If t < 0, surface is behind us — also invalid
# #         dist = jnp.where(t > 0, dist, jnp.nan)
        
# #         distances.append(dist)

# #     return jnp.array(distances)

# def meas_laser_alt(state: np.ndarray, R: np.ndarray, orientation: np.ndarray = None) -> np.ndarray:
#     """Measure laser range with additive Gaussian noise."""
#     del orientation  # TODO
#     if R.shape != (4, 4):
#         raise ValueError(f"Expected R shape (4,4), got {R.shape}")
#     return dist_from_los(state) + np.random.multivariate_normal(np.zeros(4), R)


# ####################################################################################################
# #                                       Line-of-sight Velocity
# ####################################################################################################

@jax.jit
def dist_rate_from_los(state: jnp.ndarray) -> jnp.ndarray:
    """Compute range-rate (time derivative of distance) via chain rule."""
    r, v, q, w = state[0:3], state[3:6], state[6:10], state[10:13]

    def dist_wrt_r(r_val: jnp.ndarray) -> jnp.ndarray:
        return dist_from_los(jnp.concatenate([r_val, v, q, w]))

    def dist_wrt_q(q_val: jnp.ndarray) -> jnp.ndarray:
        return dist_from_los(jnp.concatenate([r, v, q_val, w]))

    dD_dr = jax.jit(jax.jacfwd(dist_wrt_r))(r)  # (4, 3)
    dD_dq = jax.jit(jax.jacfwd(dist_wrt_q))(q)  # (4, 4)

    drdt = v
    dqdt = 0.5 * hamilton_product(q, w)

    return dD_dr @ drdt + dD_dq @ dqdt  # (4,)

# def meas_laser_vel(state: np.ndarray, R: np.ndarray, orientation: np.ndarray = None) -> np.ndarray:
#     """Measure laser range-rate with additive Gaussian noise."""
#     del orientation  # TODO
#     if R.shape != (4, 4):
#         raise ValueError(f"Expected R shape (4,4), got {R.shape}")
#     return dist_rate_from_los(state) + np.random.multivariate_normal(np.zeros(4), R)

# ####################################################################################################
# #                                       Star tracker
# ####################################################################################################

# def meas_star_tracker(q_B2L: np.ndarray, R: np.ndarray, orientation: np.ndarray = None) -> np.ndarray:
#     """Measure attitude (quaternion) with additive Gaussian noise."""
#     del orientation  # TODO
#     if R.shape != (4, 4):
#         raise ValueError(f"Expected R shape (4,4), got {R.shape}")
#     return unit(q_B2L + np.random.multivariate_normal(np.zeros(4), R))

# ####################################################################################################
# #                                       Range Tracker
# ####################################################################################################

# def meas_range_tracker(state, launchsite_pos, R: jnp.ndarray):
#     """[r, v] in inertial"""
#     r_lander = state[0:3]
#     v_lander = state[3:6]

#     print(np.array(launchsite_pos).shape)
#     print(R.shape)

#     measurements = []
#     for s in launchsite_pos:
#         rel_pos = r_lander - s[0:3]
#         rel_vel = v_lander - s[3:6]  # launch sites might be moving; subtract their vel
#         measurements.append(jnp.concatenate([rel_pos, rel_vel]))

#     return jnp.concatenate(measurements) + np.random.multivariate_normal(np.zeros(len(measurements)), R)

#     # return jnp.ravel(measurements) +


####################################################################################################
#                          NEW SENSOR ARCHITECTURE (EKF-Ready)
####################################################################################################
# This section implements a composable sensor abstraction for easy addition of sensors,
# automatic Jacobian computation, and clean EKF integration. All functions support
# time-varying sensor environments (e.g., non-stationary satellites, terrain maps).
#
# Usage:
#   1. Define a SensorEnvironment with time-varying parameters
#   2. Create Sensor objects via factory functions
#   3. Call sensor.measure(state, env) and sensor.jacobian(state, env) in EKF loop
#
# NOTE: This is new scaffolding. Measurement functions above are still used for
# backward compatibility and batch processing. Once fully migrated, batch processing
# will also use this architecture.

####################################################################################################
#                             Environment (for stuff like satellite positions)
####################################################################################################

@dataclass
class SensorEnvironment:
    """
    Holds time-varying and environmental data needed by sensors.
    Extend this dataclass to add satellite ephemeris, terrain maps, etc.
    """
    t: float  # Current time (seconds)
    mass: float
    specific_force_body: jnp.ndarray = field(default_factory=lambda: jnp.array([]))  # [3]
    satellite_positions: jnp.ndarray = field(default_factory=lambda: jnp.array([]))  # [n_sats, 3]
    satellite_velocities: jnp.ndarray = field(default_factory=lambda: jnp.array([]))  # [n_sats, 3]
    terrain_map: jnp.ndarray = field(default_factory=lambda: jnp.array([]))  # elevation grid or feature data

####################################################################################################
#                             Sensor Class
####################################################################################################

@dataclass
class Sensor:
    """
    Sensor abstraction that encapsulates measurement function, noise, and Jacobian.
    measurement_fn and noise_cov can depend on SensorEnvironment for time-varying data.
    """
    name: SensorName
    measurement_fn: Callable[[jnp.ndarray, "SensorEnvironment"], jnp.ndarray]
    noise_cov: jnp.ndarray | Callable[["SensorEnvironment"], jnp.ndarray]
    meas_dim: int

    def measure(self, state: jnp.ndarray, env: "SensorEnvironment") -> jnp.ndarray:
        """Compute measurement without noise."""
        return self.measurement_fn(state, env)

    def jacobian(self, state: jnp.ndarray, env: "SensorEnvironment") -> jnp.ndarray:
        """Compute measurement Jacobian w.r.t. state (env held fixed)."""
        return jax.jacfwd(lambda s: self.measurement_fn(s, env))(state)

    def get_noise_cov(self, env: Optional["SensorEnvironment"] = None) -> jnp.ndarray:
        """Get noise covariance matrix (may be environment-dependent)."""
        if isinstance(self.noise_cov, jnp.ndarray):
            return self.noise_cov
        else:
            return self.noise_cov(env) if env is not None else self.noise_cov()


####################################################################################################
#                             Making sensors
####################################################################################################

def accelerometer_sensor(noise_std: float) -> Sensor:
    """Accelerometer: measures body-frame specific force (inertial acceleration converted to body frame)."""
    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        # q_B2L = state[6:10]
        # specific_force_body = quat_apply(conj(q_B2L), env.specific_force_body)
        return env.specific_force_body / env.mass

    return Sensor(
        name=SensorName.ACCELEROMETER,
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(3) * (noise_std ** 2),
        meas_dim=3
    )


def gyroscope_sensor(noise_std: float) -> Sensor:
    """Gyroscope: measures body angular velocity."""
    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        del env
        return state[10:13]

    return Sensor(
        name=SensorName.GYROSCOPE,
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(3) * (noise_std ** 2),
        meas_dim=3
    )


def laser_altimeter_sensor(noise_std: float) -> Sensor:
    """Laser altimeter: range to lunar surface via LOS."""
    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        del env
        return dist_from_los(state)

    return Sensor(
        name=SensorName.LASER_ALTIMETER,
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(4) * (noise_std ** 2),
        meas_dim=4
    )


def laser_velocity_sensor(noise_std: float) -> Sensor:
    """Laser velocity: range-rate via LOS."""
    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        del env
        return dist_rate_from_los(state)

    return Sensor(
        name=SensorName.LASER_VELOCITY,
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(4) * (noise_std ** 2),
        meas_dim=4
    )


def star_tracker_sensor(noise_std: float) -> Sensor:
    """Star tracker: quaternion (attitude) measurement."""
    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        del env
        q_B2L = state[6:10]
        return unit(q_B2L)

    return Sensor(
        name=SensorName.STAR_TRACKER,
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(4) * (noise_std ** 2),
        meas_dim=4
    )


def doppler_sensor(n_sats: int, noise_std: float) -> Sensor:
    """
    Doppler sensor: measures relative velocity along line-of-sight.

    Args:
        satellite_traj_fn: Callable that takes time t and returns (r_sat, v_sat)
        noise_std: measurement noise standard deviation

    Example:
        def sat_trajectory(t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
            idx = int(t / dt)
            return sat_positions[idx], sat_velocities[idx]
        doppler = doppler_sensor(sat_trajectory, noise_std=0.1)
    """
    
    
    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        r = state[0:3]
        v = state[3:6]

        measurements = []
        for i in range(n_sats):
            r_sat = env.satellite_positions[i]
            v_sat = env.satellite_velocities[i]
            
            rel_pos = r - r_sat
            rel_vel = v - v_sat
            doppler = jnp.dot(rel_vel, rel_pos) / norm(rel_pos) # pg. 9 of that one paper (Navigation by satellite using two-way range and doppler data)
            measurements.append(doppler)
        
        return jnp.array(measurements)

    return Sensor(
        name=SensorName.DOPPLER,
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(n_sats) * (noise_std ** 2),
        meas_dim=n_sats
    )

def sat_range_tracker_sensor(n_sats: int, noise_std: float) -> Sensor:
    """Range (distance) to  (a sat currently) — directly observable position"""
    
    def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
        r_lander = state[0:3]

        measurements = []
        for i in range(n_sats):
            r_sat = env.satellite_positions[i]
            distance = norm(r_lander - r_sat)
            measurements.append(distance)
        return jnp.array(measurements)
    
    return Sensor(
        name=SensorName.DOPPLER,  # or custom RANGE
        measurement_fn=meas_fn,
        noise_cov=jnp.eye(n_sats) * (noise_std ** 2),
        meas_dim=n_sats
    )


# def terrain_relative_nav_sensor(terrain_map: jnp.ndarray, noise_std: float) -> Sensor:
#     """
#     Terrain-relative nav: measures altitude above terrain.

#     Args:
#         terrain_map: 2D elevation grid [height, width]
#         noise_std: measurement noise standard deviation
#     """
#     def meas_fn(state: jnp.ndarray, env: SensorEnvironment) -> jnp.ndarray:
#         r_lander = state[0:3]

#         ix = jnp.clip(int(r_lander[0] / 10), 0, terrain_map.shape[0] - 1)
#         iy = jnp.clip(int(r_lander[1] / 10), 0, terrain_map.shape[1] - 1)
#         z_terrain = terrain_map[ix, iy]

#         alt_above_terrain = r_lander[2] - z_terrain
#         return jnp.array([alt_above_terrain])

#     return Sensor(
#         name=SensorName.TERRAIN_RELATIVE_NAV,
#         measurement_fn=meas_fn,
#         noise_cov=jnp.array([[noise_std ** 2]]),
#         meas_dim=1
#     )


@dataclass
class SensorSuite:
    """Container for multiple sensors. Keyed by sensor name string for flexibility."""
    sensors: Dict[str, Sensor]

    def measure(self, state: jnp.ndarray, env: SensorEnvironment, sensor_name: str) -> jnp.ndarray:
        """Get measurement from a single sensor by name."""
        if sensor_name not in self.sensors:
            raise KeyError(f"Sensor '{sensor_name}' not found in suite. Available: {list(self.sensors.keys())}")
        return self.sensors[sensor_name].measure(state, env)

    def jacobian(self, state: jnp.ndarray, env: SensorEnvironment, sensor_name: str) -> jnp.ndarray:
        """Get Jacobian from a single sensor by name."""
        if sensor_name not in self.sensors:
            raise KeyError(f"Sensor '{sensor_name}' not found in suite. Available: {list(self.sensors.keys())}")
        return self.sensors[sensor_name].jacobian(state, env)

    def get_noise_cov(self, sensor_name: str, env: Optional[SensorEnvironment] = None) -> jnp.ndarray:
        """Get noise covariance from a single sensor by name."""
        if sensor_name not in self.sensors:
            raise KeyError(f"Sensor '{sensor_name}' not found in suite. Available: {list(self.sensors.keys())}")
        return self.sensors[sensor_name].get_noise_cov(env)