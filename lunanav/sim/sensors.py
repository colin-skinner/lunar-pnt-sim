import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass

from numpy.linalg import norm
from .math.quaternion import unit, conj, quat_apply

@dataclass
class Measurements:
    accel_m_s2: np.ndarray
    """3"""
    gyro_rad_s2: np.ndarray
    """3"""

class Accelerometer:

    def __init__(self, noise: jnp.ndarray = jnp.zeros(3), orientation = jnp.eye(3)):
        """Accelerometer

        Parameters
        ----------
        noise : jnp.ndarray (3,)
            
        orientation : jnp.ndarray ()
            Quaternion orientation with respect to body
        """
        self.R = jnp.diag(noise)
        self.a_body = jnp.zeros(3)

        # TODO: orientation

    @jax.jit
    def measure(self, state: jnp.ndarray, accel_sp: jnp.ndarray):
        """

        Parameters
        ----------
        state : jnp.ndarray (13,)
            (r,v,q,w)
        accel_sp : jnp.ndarray (3,)
            Specific acceleration in inertial frame (non-gravity accel)
        """        
        q_I2B = conj(state[6:10])
        self.a_body = quat_apply(q_I2B, accel_sp)

        return self.a_body

class Gyroscope:

    def __init__(self, noise: jnp.ndarray = jnp.zeros(3), orientation = jnp.eye(3)):
        """Accelerometer

        Parameters
        ----------
        noise : jnp.ndarray (3,)
            
        orientation : jnp.ndarray ()
            Quaternion orientation with respect to body
        """
        self.R = jnp.diag(noise)
        self.w_body = jnp.zeros(3)

        # TODO: orientation

    @jax.jit
    def measure(self, state: jnp.ndarray):
        """
        Parameters
        ----------
        state : jnp.ndarray (13,)
            (r,v,q,w)
        """        
        self.w_body = conj(state[10:13])

        return self.w_body

# class LaserAltimeter:


    # laser_dir_body = unit(quat_apply(q_B2I, [0,0,-1]))
    # tilt_rad = np.arccos(np.dot(r, z_body) / norm(r) / norm(z_body))
    # opposite = norm(r) * np.sin(tilt_rad)
    # x1 = opposite / np.tan(tilt_rad)
    # x0 = np.sqrt(R_MOON**2 - opposite**2)
    # alt = x1 - x0