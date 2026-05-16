import numpy as np
from numpy.linalg import norm
from dataclasses import dataclass

from .rigid_body import rigid_body_derivative, RigidBody
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

class Logger:
    def __init__(self, n_steps):
        self.t: np.ndarray = np.zeros(n_steps)
        self.X: np.ndarray = np.zeros((n_steps, 13))
        self.u: np.ndarray = np.zeros((n_steps, 6))
        self.accel_m_s2: np.ndarray = np.zeros((n_steps, 3))
        self.torque_Nm: np.ndarray = np.zeros((n_steps, 3))
        self.a_meas: np.ndarray = np.zeros((n_steps, 3))
        self.w_meas: np.ndarray = np.zeros((n_steps, 3))

    def trunc(self, n):
        vars = self.__dict__
        for varname, value in vars.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                vars[varname] = value[:n]



            
    
class Simulator:
    def __init__(self, state0: np.ndarray, t0: float, dt: float, n_steps: float):

        self.dt = dt
        self.n_steps = n_steps

        self.log = Logger(n_steps)
        self.log.X[0] = state0
        self.log.t[0] = t0

    # ----------------------------------------------------------------------
    #                       Forces and torques
    # ----------------------------------------------------------------------

    def timestep(self, t_curr: float, state: np.ndarray, mass_kg: float, I: np.ndarray, control_fn):
        """Return false if simulation is done"""

        force_N = np.zeros(3)
        torque_Nm = np.zeros(3)

        r = state[:3]
        g_moon = -GM_MOON/(norm(r)**3) * r
        force_N += g_moon * mass_kg

        # ----- Next state -----
        disturbances = np.concatenate([force_N, torque_Nm])


        # TODO: DEAL WITH ROTATION HERE AND MEASUREMENT

        # Returns body forces exerted by the spacecraft (NON gravity)
        if control_fn is not None:
            specific_force = control_fn(t_curr, state)
        else:
            specific_force = np.zeros(3)

        disturbances += specific_force
        next_state = rk4_func(0, self.dt, state, 
                              lambda t,s: rigid_body_derivative(t,s,disturbances, mass_kg, I))
        # next_state[6:10] = unit(next_state[6:10]) # to make sure quat is unitized

        # ----- Measurements -----
        w = state[10:13] # already stored in body frame because of Euler's equations
        q_B2I = unit(state[6:10]) # Turns body vector into inertial vector
        a_body = quat_apply(conj(q_B2I), specific_force[0:3] / mass_kg)

        
        
        return next_state, disturbances, Measurements(a_body, w)

    def simulate(self, control_fn = None):

        self.final_step = self.n_steps

        mass_kg = 20 # kg
        I = np.eye(3) # kg*m^2

        t = self.log.t
        X = self.log.X
        
        for step in range(self.n_steps-1):

            t_curr = t[step]
            next_step, control, meas = self.timestep(t_curr, X[step], mass_kg, I, control_fn)
            self.log.u[step] = control
            self.log.a_meas[step] = meas.accel_m_s2
            self.log.w_meas[step] = meas.gyro_rad_s2

            t[step+1] = t[step] + self.dt
            X[step+1] = next_step

            if norm(X[step+1,:]) < R_MOON:
                print("Crashed into moon")
                self.final_step = step
                break
        
        print(f"Sim stopped at step {step} / time {t[-1]:.2f}") # screw variable scoping
        self.final_step = step 

        self.log.trunc(self.final_step)



    