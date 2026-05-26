import numpy as np
from jax.numpy.linalg import norm
from dataclasses import dataclass
import jax.numpy as jnp
import jax

from .math.rigid_body import rigid_body_derivative, RigidBody
from .integrators import rk4_func
from .math.quaternion import unit, quat_apply, conj
from .sensors import Accelerometer, Gyroscope

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
    def __init__(self, state0: np.ndarray, t0: float, dt: float, n_steps: float,
                 mass_kg: float, I: jnp.ndarray):
        """Mass in kg and I in kg•m2"""

        self.dt = dt
        self.n_steps = n_steps

        self.mass = mass_kg
        self.I = I
        self.I_inv = jnp.linalg.inv(I)

        self.log = Logger(n_steps)
        self.log.X[0] = state0
        self.log.t[0] = t0

        self.accel = Accelerometer()
        self.gyro = Gyroscope()


    # ----------------------------------------------------------------------
    #                       Forces and torques
    # ----------------------------------------------------------------------

    # @jax.jit
    def rk4(self, t, state, disturbances):
        """disturbances in the """
        return rk4_func(0, self.dt, state, 
                        lambda t,s: rigid_body_derivative(t,s,disturbances, self.mass, self.I, self.I_inv))
    
    # @jax.jit
    def timestep(self, t_curr: float, state: jnp.ndarray, control_fn):
        """Timestep

        Parameters
        ----------
        t_curr : float
        state : jnp.ndarray (13,)
            _description_
        control_fn : function
            (t, state) -> (force_world, torque_body)

        Returns
        -------
        _type_
            _description_
        """

        force = jnp.zeros(3) # inertial
        torque_body = jnp.zeros(3) # body

        r, v, q_B2I, w = state[:3], state[3:6], state[6:10], state[10:13]
        g_moon = -GM_MOON/(norm(r)**3) * r
        force += g_moon * self.mass

        # ----- Next state -----
        disturbances = np.concatenate([force, torque_body])


        # TODO: DEAL WITH ROTATION HERE AND MEASUREMENT

        # Returns body forces exerted by the spacecraft (NON gravity)
        if control_fn is not None:
            specific_input = control_fn(t_curr, state)
        else:
            specific_input = np.zeros(3)

        disturbances += specific_input
        next_state = self.rk4(t_curr, state, disturbances)
        # next_state[6:10] = unit(next_state[6:10]) # to make sure quat is unitized

        # ----- Measurements -----
        self.accel.measure(next_state, specific_input[0:3]/self.mass)
        self.gyro.measure(next_state)
        
        # Laser altimeter


        
        return next_state, disturbances, Measurements(self.accel.a_body, self.gyro.w_body)

    def simulate(self, control_fn = None):

        self.final_step = self.n_steps

        t = self.log.t
        X = self.log.X
        
        for step in range(self.n_steps-1):

            t_curr = t[step]
            next_step, control, meas = self.timestep(t_curr, X[step], control_fn)
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



    