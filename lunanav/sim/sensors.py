import numpy as np
from dataclasses import dataclass

from numpy.linalg import norm
from .math.quaternion import unit, conj, quat_apply

def meas_accel(accel_body: np.ndarray, R: np.ndarray, orientation: np.ndarray = np.eye(3)):
    del orientation # TODO
    return accel_body + np.random.multivariate_normal(np.zeros(3), R)

def meas_gyro(gyro_body: np.ndarray, R: np.ndarray, orientation: np.ndarray = np.eye(3)):
    del orientation # TODO
    return gyro_body + np.random.multivariate_normal(np.zeros(3), R)
# @dataclass
# class Measurements:
#     accel_m_s2: np.ndarray
#     """3"""
#     gyro_rad_s2: np.ndarray
#     """3"""

# class Accelerometer:

#     def __init__(self, noise: np.ndarray = np.zeros(3), orientation = np.eye(3)):
#         """Accelerometer

#         Parameters
#         ----------
#         noise : np.ndarray (3,)
            
#         orientation : np.ndarray ()
#             Quaternion orientation with respect to body
#         """
#         self.R = np.diag(noise)
#         self.a_body = np.zeros(3)
#         self.orientation = orientation

#         # TODO: orientation

#     def measure(self, accel_body: np.ndarray):
#         """

#         Parameters
#         ----------
#         state : np.ndarray (13,)
#             (r,v,q,w)
#         accel_sp : np.ndarray (3,)
#             Non-gravity acceleration in body frame
#         """        
#         self.a_body = accel_body
#         return self.a_body

# class Gyroscope:

#     def __init__(self, noise: np.ndarray = np.zeros(3), orientation = np.eye(3)):
#         """Accelerometer

#         Parameters
#         ----------
#         noise : np.ndarray (3,)
            
#         orientation : np.ndarray ()
#             Quaternion orientation with respect to body
#         """
#         self.R = np.diag(noise)
#         self.w_body = np.zeros(3)
#         self.orientation = orientation

#         # TODO: orientation

#     @jax.jit
#     def measure(self, state: np.ndarray):
#         """
#         Parameters
#         ----------
#         state : np.ndarray (13,)
#             (r,v,q,w)
#         """        
#         self.w_body = state[10:13]
#         return self.w_body

# class LaserAltimeter:


    # laser_dir_body = unit(quat_apply(q_B2I, [0,0,-1]))
    # tilt_rad = np.arccos(np.dot(r, z_body) / norm(r) / norm(z_body))
    # opposite = norm(r) * np.sin(tilt_rad)
    # x1 = opposite / np.tan(tilt_rad)
    # x0 = np.sqrt(R_MOON**2 - opposite**2)
    # alt = x1 - x0