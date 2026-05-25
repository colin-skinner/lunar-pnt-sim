import sys
sys.path.append("..")

import numpy as np
# import matplotlib.pyplot as plt
from lunanav.constants import GM_MOON, R_MOON
from lunanav.sim.simulator import SimParams, SimResults, propagate, run_sim
from lunanav.plotting import debug_3d, plot_state_vector, plot_control_effort
from lunanav.sim.math.rigid_body import RigidBody
from lunanav.visualization import visualize_trajectory
import jax
import jax.numpy as jnp



def control_fn(t, state):
    """Outputs inputs in body frame"""

    force_N = np.zeros(3)
    torque_Nm = np.zeros(3)

    # ----- Thrust -----
    if t < 5:
        force_N += [0,0,3e3]

    elif t < 25:
        force_N += [0,0,8e3]

    elif t < 80:
        force_N += [0,0,30e3]

    # ----- Torque -----

    if t < 6:
        torque_Nm += [0.01, 0, 0]
    elif t < 18:
        torque_Nm += [-0.005, 0, 0]
    elif t < 36:
        torque_Nm += [0.003, 0, 0]
    elif t < 54:
        torque_Nm += [-0.003, 0, 0]

    return force_N, torque_Nm

if __name__ == "__main__":

    lander = RigidBody(
        mass_kg = 100,
        I = np.eye(3)
    )
    
    dt = .1
    t0 = 0
    t_max = 100
    nsteps = int(t_max//dt)

    state0 = np.array([
        0,0, R_MOON, 
        0 ,0, 0,
        1,0,0,0, 
        0,0,0])

    sim = SimParams(state0, lander, dt, nsteps)
    results = run_sim(state0, t_max, dt, control_fn, sim)
    n = results.nsteps

    
    # plot_state_vector(results.t,
    #                   r = results.states[:, 0:3] - np.tile([0,0,R_MOON], (n, 1)),
    #                   v = results.states[:, 3:6],
    #                   w = results.states[:, 10:13])
    
    # plot_control_effort(results.t, results.force_N, results.torque_Nm)

    moon_offset =  np.tile([0,0,R_MOON,0,0,0,0,0,0,0,0,0,0], (n, 1))

    fig = visualize_trajectory(results.states - moon_offset, results.t, dt)
    fig.show()

    print("done?!")
    
    

    # # Plot in 3D
    # # limits = np.array([
    # #     [-25,25],
    # #     [-20, 30],
    # #     [1740, 1790]
    # # ])

    # limits = np.array([
    #     [-300,300],
    #     [-290, 10],
    #     [0, 300]
    # ])

    # launch_centered_X = log.X
    # launch_centered_X[:, 0:3] -= state0[0:3]

    # debug_3d(launch_centered_X, log.t, dt, limits = limits)


