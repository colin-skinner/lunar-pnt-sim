# ruff: noqa: E741
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .quaternion import hamilton_product

class RigidBodyParams:
    """Add force with add_force() or add_torque()"""

    def __init__(self, mass_kg, I):
        self.mass_kg = mass_kg
        self.I = I
        self.force_N = np.zeros(3)
        self.torque_Nm   = np.zeros(3)

    def add_force(self, force: np.ndarray, r: np.ndarray = None, verbose = False):
        """Adds force in the global frame of the drone"""

        if np.shape(force) != (3,):
            raise ValueError("force must be a 3x1 vector")

        if r is not None and np.shape(r) != (3,):
            raise ValueError("r_body must be a 3x1 vector")

        self.force_N += force

        if r is not None:
            torque = np.cross(r, force)
            self.torque_Nm += torque

        if verbose:
            print(f"Rigid Body Add: {r=} {force=} {torque=}")

    def add_torque(self, torque: np.ndarray):
        """Adds force in the global frame of the drone"""

        torque = np.asarray(torque)

        if np.shape(torque) != (3,):
            raise ValueError("torque must be a 3x1 vector")
        self.torque_Nm += torque

def rigid_body_derivative(t: float, state: np.ndarray, params: RigidBodyParams):
    v = state[3:6]
    q = state[6:10]
    w = state[10:13]

    # Position derivative is velocity 
    drdt = v

    # Velocity derivative is acceleration (Schaub 2.15)
    dvdt = np.asarray(params.force_N) / params.mass_kg

    # Quaternion derivative is based on hamilton product (Schaub 3.111)
    dqdt = 0.5 * hamilton_product(q, w)
    # Angular derivative based on (Schaub 4.34-35)

    I = params.I
    if I is not None:
        I_inv = np.linalg.inv(I) #TODO: maybe precalc this
        torque = np.asarray(params.torque_Nm)

        # τ = parameters.torque_body
        dwdt = I_inv @ (torque - np.cross(w, I @ w))
    else:
        dwdt = np.zeros(3)

    return np.hstack((drdt, dvdt, dqdt, dwdt))
