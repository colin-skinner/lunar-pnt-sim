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
def dist_from_los(state, altimeter_bound_m = 100e3, min_dist_inside_moon = 10e3):
    """
    Find intersection of LOS rays with lunar sphere.
    LOS ray: r + t * d, where d is LOS direction in inertial frame.
    Sphere: |x| = R_MOON.
    
    Solve: |r + t*d|^2 = R_MOON^2
    Quadratic in t: t^2 + 2*(r·d)*t + |r|^2 - R_MOON^2 = 0
    """
    r, q_B2L = state[0:3], state[6:10]
    r_norm = norm(r)
    vecs_body = get_los_vectors()
    distances = []

    for v in vecs_body:
        d = quat_apply(q_B2L, v)
        d = unit(d)
        r_along_d = jnp.dot(r,d) / norm(d) # negative if "opposite direction"
        cos_angle = r_along_d / r_norm
        sin_angle = jnp.sqrt(1 - cos_angle**2)
        r_perp_d = r_norm * sin_angle

        dist_inside_moon_2 = R_MOON**2 - r_perp_d**2

        LOS_dist = abs(r_along_d) - jnp.sqrt(dist_inside_moon_2)

        # If closest point is inside moon (with bound)
        distance = jnp.where(dist_inside_moon_2 > min_dist_inside_moon**2, LOS_dist, jnp.nan)

        # If point is even in moon direction and not behind lander
        distance = jnp.where(r_along_d < 0,    distance,     jnp.nan)

        # Max bound
        distance = jnp.where(LOS_dist <= altimeter_bound_m, distance, jnp.nan)


        distances.append(distance)
    

    return jnp.array(distances)


# ####################################################################################################
# #                                       Line-of-sight Velocity
# ####################################################################################################

@jax.jit
def dist_rate_from_los(state: jnp.ndarray) -> jnp.ndarray:
    """Compute range-rate (time derivative of distance) via chain rule."""
    r, v, q, w = state[0:3], state[3:6], state[6:10], state[10:13]

    mask = jnp.isnan(dist_from_los(state))

    def dist_wrt_r(r_val: jnp.ndarray) -> jnp.ndarray:
        return dist_from_los(jnp.concatenate([r_val, v, q, w]))

    def dist_wrt_q(q_val: jnp.ndarray) -> jnp.ndarray:
        return dist_from_los(jnp.concatenate([r, v, q_val, w]))

    dD_dr = jax.jit(jax.jacfwd(dist_wrt_r))(r)  # (4, 3)
    dD_dq = jax.jit(jax.jacfwd(dist_wrt_q))(q)  # (4, 4)

    drdt = v
    dqdt = 0.5 * hamilton_product(q, w)

    vels = dD_dr @ drdt + dD_dq @ dqdt
    vels = jnp.where(mask, jnp.nan, vels)

    return vels  # (4,)


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