import numpy as np
from jax.numpy.linalg import norm
from dataclasses import dataclass, field
import jax.numpy as jnp
import jax

from .math.rigid_body import rigid_body_derivative, RigidBody
from .math.integration import rk4_func
from .math.quaternion import unit, quat_apply, conj
from .sensors import meas_accel, meas_gyro

from ..constants import GM_MOON, R_MOON

class SimResults:
    def __init__(self, n_steps):
        self.t: np.ndarray = np.zeros(n_steps)
        self.states: np.ndarray = np.zeros((n_steps, 13))
        self.u: np.ndarray = np.zeros((n_steps, 6))
        self.force_N: np.ndarray = np.zeros((n_steps, 3))
        self.torque_Nm: np.ndarray = np.zeros((n_steps, 3))
        self.a_meas: np.ndarray = np.zeros((n_steps, 3))
        self.w_meas: np.ndarray = np.zeros((n_steps, 3))

    def trunc(self, n):
        vars = self.__dict__
        for varname, value in vars.items():
            if isinstance(value, (np.ndarray)):
                vars[varname] = value[:n]
            else:
                raise ValueError(f"Should only have np.ndarray fields, but {varname} is {type(value)}")
            
        self.nsteps = n


@dataclass
class SimParams:

    """Simulation parameters. Mass in kg and I in kg•m2"""
    state0: float
    body: RigidBody = field(default_factory=RigidBody)
    dt: float = 0.1  # time step (seconds)
    t_end: float = 100.0  # max simulation time
    sensor_noises: dict = field(default_factory=dict) # dict of sensor name to noise covariance

####################################################################################################
#                                       O
####################################################################################################

def propagate(state, force_body, torque_body, params: SimParams, mu: float = GM_MOON):
    """
    Propagate the state of the rigid body forward in time.

    Parameters
    ----------
    state : jnp.ndarray (13,)
        Initial state vector [r, v, q, w]
    force : jnp.ndarray (3,)
        Force acting on the body (in body frame) `[N]`
    torque : jnp.ndarray (3,)
        Torque acting on the body (in body frame) `[Nm]`
    params : SimParams
    mu : float, optional
        Gravitational parameter (GM) of the central body `[m3/s2]`. If 0, no gravity forces are applied.

    Returns
    -------
    jnp.ndarray
        Next state vector [r, v, q, w]
    """

    q_B2L = state[6:10]
    force = quat_apply(q_B2L, force_body) # body force --> inertial force

    if mu > 0:
        r = state[0:3]
        force += -mu * r / norm(r)**3

    def state_dot(t, s):
        return rigid_body_derivative(t, s, force, torque_body, params.body.mass_kg, params.body.I)

    return rk4_func(0, params.dt, state, state_dot)

def run_sim(state0, nsteps, dt, control_fn, params: SimParams):
    """Run the simulation forward in time.

    Parameters
    ----------
    state0 : jnp.ndarray (13,)
        Initial state vector [r, v, q, w]
    tmax : float
        Maximum simulation time
    dt : float
        Time step for integration
    control_fn : function
        Control function that takes in (t, state) and outputs (force_body, torque_body)
    params : SimParams

    Returns
    -------
    SimLogger
        Logger containing the history of states and controls.
    """

    final_step = nsteps # for when the sim ends and we truncate
    logger = SimResults(nsteps)

    if norm(state0[0:3]) < 1e-3:
        raise ValueError("Initial position is [0,0,0]")
    logger.states[0] = state0

    final_step = nsteps
    for step in range(nsteps-1):
        t_curr = logger.t[step]
        state = logger.states[step]

        force_body, torque_body = control_fn(t_curr, state)
        logger.force_N[step] = force_body
        logger.torque_Nm[step] = torque_body


        logger.a_meas[step] = meas_accel(force_body/params.body.mass_kg, params.sensor_noises.get("accel", np.zeros((3,3))))
        logger.w_meas[step] = meas_gyro(state[10:13], params.sensor_noises.get("gyro", np.zeros((3,3))))


        next_state = propagate(state, force_body, torque_body, params)
        logger.t[step+1] = logger.t[step] + dt
        logger.states[step+1] = next_state

        if norm(logger.states[0:3]) < R_MOON:
            print("Crashed into moon")
            final_step = step
            break

        
    logger.trunc(final_step)

    logger.a_meas = logger.a_meas[:-1]
    logger.w_meas = logger.w_meas[:-1]

    return logger