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
def rigid_body_derivative(t: float, state: jnp.ndarray, disturbances: jnp.ndarray, mass_kg: float, I: np.ndarray):
    v = state[3:6]
    q_B2I = state[6:10]
    w = state[10:13]

    force_N = disturbances[0:3]
    torque_Nm = disturbances[3:6]

    # Position derivative is velocity 
    drdt = v

    # Velocity derivative is acceleration (Schaub 2.15)
    dvdt = jnp.asarray(force_N) / mass_kg

    # Quaternion derivative is based on hamilton product (Schaub 3.111)
    dqdt = 0.5 * hamilton_product(q_B2I, w)
    # Angular derivative based on (Schaub 4.34-35)

    I = I
    if I is not None:
        I_inv = jnp.linalg.inv(I) #TODO: maybe precalc this
        torque = jnp.asarray(torque_Nm)

        # τ = parameters.torque_body
        dwdt = I_inv @ (torque - jnp.cross(w, I @ w))
    else:
        dwdt = jnp.zeros(3)

    return jnp.hstack((drdt, dvdt, dqdt, dwdt))
