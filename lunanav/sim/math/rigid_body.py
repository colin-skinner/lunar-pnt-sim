# ruff: noqa: E741
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

import jax.numpy as jnp
import jax

from .quaternion import hamilton_product

@dataclass
class RigidBody:
    mass_kg: float
    I: float

@jax.jit
def rigid_body_derivative(t: float, state: jnp.ndarray, force: jnp.ndarray, torque: jnp.ndarray,
                          mass_kg: float, I: jnp.ndarray):
    """Rigid Body dynamics (can be wrapped for optimizer).

    Parameters
    ----------
    t : float
        current time
    state : jnp.ndarray (13,)
        x, v, q, w 
    force : jnp.ndarray (3,)
        Force (inertial)
    torque : jnp.ndarray (3,)
        Torque (body)
    mass_kg : float
    I : jnp.ndarray
        Moment of inertia (body frame)

    Returns
    -------
    jnp.ndarray
        State derivative (13,)
    """
    del t
    
    # r,v inertial, w in body frame 
    v, q_B2I, w = state[3:6], state[6:10], state[10:13]

    # Inertial frame

    # Position derivative is velocity 
    drdt = v

    # Velocity derivative is acceleration (Schaub 2.15)
    dvdt = jnp.asarray(force) / mass_kg

    # Quaternion derivative is based on hamilton product (Schaub 3.111)
    dqdt = 0.5 * hamilton_product(q_B2I, w)
    # print(dqdt)
    # print(0.5 * np.array([
    #     [ 0,    -w[0], -w[1], -w[2]],
    #     [ w[0],  0,     w[2], -w[1]],
    #     [ w[1], -w[2],  0,     w[0]],
    #     [ w[2],  w[1], -w[0],  0  ],
    # ]) @ q_B2I)
    # breakpoint()

    # Angular derivative (Schaub 4.34-35)
    Tau = jnp.asarray(torque)
    dwdt = jnp.linalg.inv(I) @ (Tau - jnp.cross(w, I @ w))

    return jnp.concatenate([drdt, dvdt, dqdt, dwdt])
