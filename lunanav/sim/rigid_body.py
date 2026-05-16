# ruff: noqa: E741
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

import jax.numpy as jnp
import jax

from .quaternion import hamilton_product

class RigidBody:

    def __init__(self, mass_kg: float, I: jnp.ndarray):
        self.mass_kg = mass_kg
        self.I = I
        self.force_N = np.zeros(3)
        self.torque_Nm   = np.zeros(3)

@jax.jit
def rigid_body_derivative(t: float, state: jnp.ndarray, disturbances: jnp.ndarray,
                          mass_kg: float, I: np.ndarray, I_inv: np.ndarray):
    """Rigid Body dynamics (can be wrapped for optimizer)

    Parameters
    ----------
    t : float
        current time
    state : jnp.ndarray (13,)
        x, v, q, w 
    disturbances : jnp.ndarray (6,)
        force, torque
    mass_kg : float
        _description_
    I : np.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """
    
    # r,v inertial, w in body frame 
    v, q_B2I, w = state[3:6], state[6:10], state[10:13]

    # Inertial frame
    F, Tau = disturbances[0:3], disturbances[3:6]

    # Position derivative is velocity 
    drdt = v

    # Velocity derivative is acceleration (Schaub 2.15)
    dvdt = jnp.asarray(F) / mass_kg

    # Quaternion derivative is based on hamilton product (Schaub 3.111)
    dqdt = 0.5 * hamilton_product(q_B2I, w)
    print(dqdt)
    print(0.5 * np.array([
        [ 0,    -w[0], -w[1], -w[2]],
        [ w[0],  0,     w[2], -w[1]],
        [ w[1], -w[2],  0,     w[0]],
        [ w[2],  w[1], -w[0],  0  ],
    ]) @ q_B2I)
    breakpoint()

    # Angular derivative (Schaub 4.34-35)
    Tau = jnp.asarray(Tau)
    dwdt = I_inv @ (Tau - jnp.cross(w, I @ w))

    return jnp.hstack((drdt, dvdt, dqdt, dwdt))
