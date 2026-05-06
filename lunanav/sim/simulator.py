import numpy as np
from numpy.linalg import norm

from .rigid_body import RigidBodyParams, rigid_body_derivative
from .integrators import rk4_func
from .quaternion import unit, quat_apply

from ..constants import GM_MOON

# `from ..entities import Spacecraft
# from ..math import rk4_func, quat_apply, unit
# from ..physics.rigid_body import rigid_body_derivative, RigidBodyParams
# from ..physics import grav_accel
# from ..world import MU_EARTH
# from ..physics.energy import calc_potential_energy, calc_kinetic_energy`
# TODO: def finish this

# TODO: Different integrators for better ECI-ECEF OR interpolate every second or something

class Sim6DOF:
    def __init__(self, state0: np.ndarray, t0: float, dt: float, n_steps: float):

        self.t = t0
        self.dt = dt
        self.n_steps = n_steps

        self.X = np.zeros((n_steps+1, 13))
        self.X[0] = state0

    def step_one(self, state: np.ndarray, rigid_body_params: RigidBodyParams):
        """Return false if simulation is done"""

        # Get measured angular velocity
        # w = state[10:13]
        # q_B2I = unit(state[6:10]) # Turns body vector into inertial vector

        # w_body = quat_apply(q_B2I, w)


        # Simulate next state
        r = state[:3]
        g_moon = -GM_MOON/(norm(r)**3) * r
        rigid_body_params.force_N = g_moon * rigid_body_params.mass_kg


        # Propagate 1 step
        next_state = rk4_func(self.t, self.dt, state, rigid_body_derivative, rigid_body_params)

        next_state[6:10] = unit(next_state[6:10])
        
        return next_state

    def simulate(self):

        self.final_step = self.n_steps

        mass_kg = 5 # kg
        torque = np.zeros(3) # Nm
        I = np.eye(3) # kg*m^2
        
        for step in range(self.n_steps):

            next_step = self.step_one(self.X[step], RigidBodyParams(mass_kg, torque_Nm = torque, I = I))
            self.t += self.dt

            self.X[step+1] = next_step

        self.X = self.X[:self.final_step]


    