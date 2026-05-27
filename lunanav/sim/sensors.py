import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field

from jax.numpy.linalg import norm
from .quaternion import quat_apply, angle_axis_to_q
from ..constants import R_MOON


@dataclass
class SensorNoises:
    accel: np.ndarray = field(default_factory=lambda: np.zeros((3,3)))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros((3,3)))
    laser_alt: np.ndarray = field(default_factory=lambda: np.zeros((4,4)))

####################################################################################################
#                                       Accel
####################################################################################################

def meas_accel(accel_body: np.ndarray, R: np.ndarray, orientation: np.ndarray = np.eye(3)):
    del orientation # TODO
    return accel_body + np.random.multivariate_normal(np.zeros(3), R)

####################################################################################################
#                                       Gyro
####################################################################################################

def meas_gyro(gyro_body: np.ndarray, R: np.ndarray, orientation: np.ndarray = np.eye(3)):
    del orientation # TODO
    return gyro_body + np.random.multivariate_normal(np.zeros(3), R)

####################################################################################################
#                                       Los
####################################################################################################

los_vectors = np.array([
    quat_apply(angle_axis_to_q(135, [-1,1,0], degrees=True), [0,0,1]),
    quat_apply(angle_axis_to_q(135, [-1,-1,0], degrees=True), [0,0,1]),
    quat_apply(angle_axis_to_q(135, [1,-1,0], degrees=True), [0,0,1]),
    quat_apply(angle_axis_to_q(135, [1,1,0], degrees=True), [0,0,1])
])

def get_los_vectors():
    """M: sensor frame"""
    return los_vectors # already calculated when module is imported, so only calculated once

def alt_from_los(state):
    r, q_B2L = state[0:3], state[6:10]
    vecs_body = get_los_vectors() 
    distances = []

    for v in vecs_body:
        vec_inertial = quat_apply(q_B2L, v) # in 
        tilt_rad = jnp.arccos(jnp.dot(r, vec_inertial) / norm(r) / norm(vec_inertial))
        alt = norm(r) - R_MOON
        dist = alt / jnp.cos(tilt_rad)
        distances.append(dist)

    return jnp.array(distances)

"""Could be a better one"""
# def alt_from_los(state):
#     """
#     Find intersection of LOS rays with lunar sphere.
#     LOS ray: r + t * d, where d is LOS direction in inertial frame.
#     Sphere: |x| = R_MOON.
    
#     Solve: |r + t*d|^2 = R_MOON^2
#     Quadratic in t: t^2 + 2*(r·d)*t + |r|^2 - R_MOON^2 = 0
#     """
#     r, q_B2L = state[0:3], state[6:10]
#     vecs_body = get_los_vectors()
#     distances = []

#     for v in vecs_body:
#         d = quat_apply(q_B2L, v)
#         d = d / jnp.linalg.norm(d)  # ensure unit vector
        
#         # Quadratic coefficients
#         b = jnp.dot(r, d)
#         c = jnp.dot(r, r) - R_MOON**2
        
#         discriminant = b**2 - c
        
#         # Two solutions: t = -b ± sqrt(discriminant)
#         # Pick the smaller positive one (closer intersection)
#         t = -b - jnp.sqrt(jnp.maximum(discriminant, 0))
        
#         # If discriminant < 0, no intersection (LOS misses surface)
#         dist = jnp.where(discriminant > 0, t, jnp.nan)
#         # If t < 0, surface is behind us — also invalid
#         dist = jnp.where(t > 0, dist, jnp.nan)
        
#         distances.append(dist)

#     return jnp.array(distances)

def meas_laser_alt(state: np.ndarray, R: np.ndarray, orientation: np.ndarray = np.eye(3)):
    del orientation # TODO
    return alt_from_los(state) + np.random.multivariate_normal(np.zeros(4), R)