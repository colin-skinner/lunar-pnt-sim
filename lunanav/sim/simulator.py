from typing import Callable
import numpy as np
from jax.numpy.linalg import norm
from dataclasses import dataclass, field
import jax.numpy as jnp
import jax

from .quaternion import unit, quat_apply, conj, hamilton_product, unitize_state
from .sensors import meas_accel, meas_gyro, meas_laser_alt, meas_laser_vel, meas_star_trackcer, SensorNoises

from ..constants import GM_MOON, R_MOON

@dataclass
class RigidBody:
    mass_kg: float
    I: float

class SimResults:
    def __init__(self, n_steps):
        self.t: np.ndarray = np.zeros(n_steps)
        self.states: np.ndarray = np.zeros((n_steps, 13))
        self.u: np.ndarray = np.zeros((n_steps, 6))
        self.force_N: np.ndarray = np.zeros((n_steps, 3))
        self.torque_Nm: np.ndarray = np.zeros((n_steps, 3))

    def trunc(self, n):
        vars = self.__dict__
        for varname, value in vars.items():
            if isinstance(value, (np.ndarray)):
                vars[varname] = value[:n]
            else:
                raise ValueError(f"Should only have np.ndarray fields, but {varname} is {type(value)}")
            
        self.nsteps = n

@dataclass
class SimMeasurements:
    accel: np.ndarray
    gyro: np.ndarray
    laser_alt: np.ndarray
    laser_vel: np.ndarray
    star_tracker: np.ndarray


@dataclass
class SimParams:

    """Simulation parameters. Mass in kg and I in kg•m2"""
    state0: float
    body: RigidBody = field(default_factory=RigidBody)
    dt: float = 0.1  # time step (seconds)
    t_end: float = 100.0  # max simulation time

####################################################################################################
#                                       Rigid and motion
####################################################################################################

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

    assert state.shape == (13,)
    assert force.shape == (3,)
    assert torque.shape == (3,)
    assert I.shape == (3,3)
    
    # r,v inertial, w in body frame 
    v, q_B2I, w = state[3:6], state[6:10], state[10:13]

    # Position derivative is velocity 
    drdt = v
    dvdt = jnp.asarray(force) / mass_kg
    dqdt = 0.5 * hamilton_product(q_B2I, w)

    # Angular derivative (Schaub 4.34-35)
    Tau = jnp.asarray(torque)
    dwdt = jnp.linalg.inv(I) @ (Tau - jnp.cross(w, I @ w))

    return jnp.concatenate([drdt, dvdt, dqdt, dwdt])

@jax.jit
def rk4_next_step(t: float, dt: float, state_prev: float, force_I: jnp.ndarray, torque_B: jnp.ndarray, mass: float, I_B: jnp.ndarray):

    def helper(t,s):
        return rigid_body_derivative(t,s,force_I, torque_B, mass, I_B) 
    
    k1 = helper(t, state_prev)
    k2 = helper(t + dt / 2, state_prev + dt * k1 / 2)
    k3 = helper(t + dt / 2, state_prev + dt * k2 / 2)
    k4 = helper(t + dt, state_prev + dt * k3)

    x_n = state_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_n

####################################################################################################
#                                       Actual propagation
####################################################################################################

@jax.jit
def lander_motion(state: jnp.ndarray, force_B: jnp.ndarray, torque_B: jnp.ndarray, dt: float, mass: float, I: np.ndarray, mu: float = GM_MOON):
    """
    Propagate the state of lander forward in time with gravity

    Parameters
    ----------
    state : jnp.ndarray (13,)
        Initial state vector [r, v, q, w]
    force : jnp.ndarray (3,)
        Force acting on the body (in body frame) `[N]`
    torque_B : jnp.ndarray (3,)
        Torque acting on the body (in body frame) `[Nm]`
    params : SimParams
    mu : float, optional
        Gravitational parameter (GM) of the central body `[m3/s2]`. If 0, no gravity forces are applied.

    Returns
    -------
    jnp.ndarray
        Next state vector [r, v, q, w]
    """

    q_B2L = unit(state[6:10])
    
    force_I = quat_apply(q_B2L, force_B) # body force --> inertial force

    # Add gravity if need be
    r = state[0:3]
    force_I = jnp.where(mu > 0,
                        force_I - mu * r / norm(r)**3 * mass,
                        force_I)

    next_state = rk4_next_step(0, dt, state, force_I, torque_B, mass, I)
    next_state = unitize_state(next_state)

    return next_state

def linearized_lander_motion(state: jnp.ndarray, force_B: jnp.ndarray, torque_B: jnp.ndarray,
                          dt: float, mass: float, I: jnp.ndarray):
    """Linearize propagation under gravity around a state and BODY input."""
    return jax.jacfwd(lambda s, f, tau: lander_motion(s, f, tau, dt, mass, I))(state, force_B, torque_B)

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

        next_state = lander_motion(state, force_body, torque_body, params.dt, params.body.mass_kg, params.body.I)

        if np.any(np.isnan(next_state)):
            print(f"NaN at step {step}, t={t_curr}")
            print(f"prev state: {state}")
            print(f"force_body: {force_body}")
            print(f"next_state: {next_state}")
            break
        logger.t[step+1] = logger.t[step] + dt
        logger.states[step+1] = next_state

        if norm(next_state[0:3]) < R_MOON:
            print("Crashed into moon")
            final_step = step
            break

        
    logger.trunc(final_step)

    return logger

def calc_measurements(results: SimResults, mass: float, sensor_noises: SensorNoises = SensorNoises()):
    states = results.states
    forces = results.force_N

    accel = np.array([meas_accel(force / mass, sensor_noises.accel, state[6:10]) for force, state in zip(forces, states)])
    print("Accel done")
    gyro = np.array([meas_gyro(state[10:13], sensor_noises.gyro, state[6:10]) for state in states])
    print("Gyro done")
    laser_alt = np.array([meas_laser_alt(state, sensor_noises.laser_alt, state[6:10]) for state in states])
    print("Laser dist done")
    laser_vel = np.array([meas_laser_vel(state, sensor_noises.laser_vel, state[6:10]) for state in states])
    print("Laser vel done")
    q_star_tracker = np.array([meas_star_trackcer(state[6:10], sensor_noises.star_tracker) for state in states])
    print("Quat done")
    measurements = SimMeasurements(accel, gyro, laser_alt, laser_vel, q_star_tracker)
    
    return measurements