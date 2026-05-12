import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass

from .rigid_body import RigidBodyParams, rigid_body_derivative
from .integrators import rk4_func
from .quaternion import unit, quat_apply, conj

from ..constants import GM_MOON, R_MOON

# `from ..entities import Spacecraft
# from ..math import rk4_func, quat_apply, unit
# from ..physics.rigid_body import rigid_body_derivative, RigidBodyParams
# from ..physics import grav_accel
# from ..world import MU_EARTH
# from ..physics.energy import calc_potential_energy, calc_kinetic_energy`
@dataclass
class Measurements:
    accel_m_s2: np.ndarray
    """3"""
    gyro_rad_s2: np.ndarray
    """3"""
    
class Sim6DOF:
    def __init__(self, state0: np.ndarray, t0: float, dt: float, n_steps: float):

        self.dt = dt
        self.n_steps = n_steps

        self.t = np.zeros(n_steps+1)
        self.X = np.zeros((n_steps+1, 13))
        self.a_meas = np.zeros((n_steps+1, 3))
        self.w_meas = np.zeros((n_steps+1, 3))
        self.X[0] = state0
        self.t[0] = t0

    # ----------------------------------------------------------------------
    #                       Forces and torques
    # ----------------------------------------------------------------------




    def motion_step(self, state: np.ndarray, rigid_body_params: RigidBodyParams):
        """Return false if simulation is done"""

        status = "good" 

        # ----- Forces and torques (are 0 in params) -----
        assert np.array_equal(rigid_body_params.force_N, [0,0,0])
        assert np.array_equal(rigid_body_params.torque_Nm, [0,0,0])

        # Moon gravity
        r = state[:3]
        g_moon = -GM_MOON/(norm(r)**3) * r
        rigid_body_params.add_force(g_moon * rigid_body_params.mass_kg)

        # Thrust
        q_B2I = unit(state[6:10])
        
        thrust = [0,0,5] # in body [N]
        thrust = quat_apply(q_B2I, thrust)
        rigid_body_params.add_force(thrust)

        # ----- Next state -----
        next_state = rk4_func(self.t, self.dt, state, rigid_body_derivative, rigid_body_params)
        next_state[6:10] = unit(next_state[6:10]) # to make sure quat is unitized
        
        # Get measured angular velocity
        w = state[10:13]
        q_B2I = unit(state[6:10]) # Turns body vector into inertial vector
        w_body = quat_apply(q_B2I, w)

        a = rigid_body_params.force_N/rigid_body_params.mass_kg
        a_body = quat_apply(conj(q_B2I), a)

        
        
        return next_state, status,  Measurements(a_body, w_body)

    def simulate(self):

        self.final_step = self.n_steps

        mass_kg = 20 # kg
        I = np.eye(3) # kg*m^2
        
        for step in range(self.n_steps):

            next_step, status, meas = self.motion_step(self.X[step], RigidBodyParams(mass_kg, I = I))
            self.a_meas[step] = meas.accel_m_s2
            self.w_meas[step] = meas.gyro_rad_s2

            self.t[step+1] = self.t[step] + self.dt
            self.X[step+1] = next_step

            if norm(self.X[step+1,:]) < R_MOON:
                print("Crashed into moon")
                self.final_step = step
                break

            if status != "good":
                print(status)
                break


        # Truncate things
        self.X = self.X[:self.final_step]
        self.t = self.t[:self.final_step]
        self.a_meas = self.a_meas[:self.final_step]
        self.w_meas = self.w_meas[:self.final_step]
        print(f"Sim stopped at step {step} / time {self.t[-1]}") # screw variable scoping


    