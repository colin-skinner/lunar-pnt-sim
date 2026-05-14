import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from lunanav.constants import GM_MOON, R_MOON
from lunanav.sim.simulator import Simulator
from lunanav.plotting import debug_3d, plot_state_vector, plot_3
from lunanav.sim.rigid_body import rigid_body_derivative, RigidBody
from lunanav.sim.quaternion import quat_apply, unit, conj, hamilton_product
import jax




def control_fn(t, state):
    """Outputs force """

    force_N = np.zeros(3)
    torque_Nm = np.zeros(3)

    q_B2I = unit(state[6:10])

    # ----- Thrust -----
    if t < 5:
        force_N += quat_apply(q_B2I, [0,0,3])

    elif t < 25:
        force_N += quat_apply(q_B2I, [0,0,8])

    elif t < 80:
        force_N += quat_apply(q_B2I, [0,0,30])

    # ----- Torque -----

    if t < 6:
        torque_Nm += [0.01, 0, 0]
    elif t < 18:
        torque_Nm += [-0.005, 0, 0]
    elif t < 36:
        torque_Nm += [0.003, 0, 0]
    elif t < 54:
        torque_Nm += [-0.003, 0, 0]

    # mass_kg = 20 # kg
    # I = np.eye(3) # kg*m^2
    # print(np.diag(torque_Nm*0.1))

    # force_N += np.random.multivariate_normal(np.zeros(3), np.diag(force_N*0.02))
    # torque_Nm += np.random.multivariate_normal(np.zeros(3), np.diag(torque_Nm*0.02))
    # [r]

    return np.concatenate([force_N, torque_Nm])

if __name__ == "__main__":

    r = R_MOON + 15 # GEO
    v = np.sqrt(GM_MOON/r) # circular velocity

    dt = .1
    t0 = 0
    t_max = 100
    n_steps = int(t_max//dt)

    state0 = np.array([
        0,0, r, 
        0 ,0, 0,
        1,0,0,0, 
        0,0,0])
    
    control0 = np.ones(6)

    sim = Simulator(state0, t0, dt, n_steps)
    sim.simulate(control_fn)
    log = sim.log


    jac = jax.jacfwd(rigid_body_derivative, argnums=(1,2))  # 13×13
    # print(jac(0, log.X[-4], log.u[-4], RigidBody(5,np.eye(3))))

    # plot_3(log.t, log.u)

    # Plot states
    plot_state_vector(log.t, log.X)
    plt.show()
    

    # Plot in 3D
    # limits = np.array([
    #     [-25,25],
    #     [-20, 30],
    #     [1740, 1790]
    # ])

    limits = np.array([
        [-300,300],
        [-290, 10],
        [0, 300]
    ])

    launch_centered_X = log.X
    launch_centered_X[:, 0:3] -= state0[0:3]

    debug_3d(launch_centered_X, log.t, dt, limits = limits)


