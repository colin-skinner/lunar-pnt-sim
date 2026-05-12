import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
from lunanav.constants import GM_MOON, R_MOON
from lunanav.sim.simulator import Sim6DOF
from lunanav.plotting import debug_3d, plot_state_vector, plot_3


if __name__ == "__main__":

    r = R_MOON + 15 # GEO
    v = np.sqrt(GM_MOON/r) # circular velocity
    print(r)
    print(v)


    dt = .1
    t0 = 0
    n_steps = 200 # About 2 days

    state0 = np.array([
        0,0, r, 
        0 ,0, 0,
        1,0,0,0, 
        1,0,0])

    sim = Sim6DOF(state0, t0, dt, n_steps)

    sim.simulate()


    plot_3(sim.t, sim.a_meas)
    plot_3(sim.t, sim.w_meas)


    # print(sim.X)
    # print(sim.t)

    # plt.plot(sim.X)

    # plot_state_vector(sim.t, sim.X)
    # plt.show()
    
    limits = np.array([
        [-100,100],
        [-50, 150],
        [1740, 1940]
    ])

    debug_3d(sim.X, sim.t, dt, limits = limits)


