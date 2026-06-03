import time

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np

from scipy.integrate import odeint

import sys
sys.path.append("..")

from lunanav.visualization import visualize_trajectory
from lunanav.constants import R_MOON

if __name__ == "__main__":
    # Example: create dummy data and visualize
    np.random.seed(42)
    n_steps = 300
    dt = 0.1
    
    # Dummy descent trajectory
    t = np.arange(n_steps) * dt
    z0 = 10 + R_MOON # km
    vz0 = -1.5 # km/s
    accel = 4.5  # m/s^2
    
    r = np.zeros((n_steps, 3))
    v = np.zeros((n_steps, 3))
    q = np.zeros((n_steps, 4))
    w = np.zeros((n_steps, 3))
    
    for i in range(n_steps):
        r[i, 2] = z0 + vz0 * t[i] + 0.5 * accel * t[i]**2
        v[i, 2] = vz0 + accel * t[i]
        q[i] = np.array([1, 0, 0, 0])  # identity quat
        w[i] = np.zeros(3)
    
    states = np.hstack([r, v, q, w])
    forces = np.zeros((n_steps, 6))
    forces[:, 2] = 1500 * 9.81  # constant thrust
    
    # Visualize
    fig = visualize_trajectory(states, forces, t, dt)
    fig.show()
    # fig.write_html("trajectory.html")  # uncomment to save