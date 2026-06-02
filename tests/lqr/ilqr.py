import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm

import sys
sys.path.append("../..")

from lunanav.sim.dynamics import lander_motion, linearize, lander_motion_inertial
from lunanav.sim.quaternion import angle_axis_to_q
from lunanav.plotting import plot_state_vector, debug_3d, plot_control_effort
from lunanav.constants import R_MOON, GM_MOON, RAD_TO_DEG, DEG_TO_RAD
from lunanav.visualization import visualize_trajectory
from lunanav.control.lqr import ilqr_lander

def get_rv_costs():
    Q = np.diag([
        1e-3, 1e-3, 1e-3,    # position (m) — moderate
        1e-0, 1e-0, 1e-0,    # velocity (m/s) — higher (you want soft touchdown)

        1e-1, 1e-1, 1e-1, 1e-1,  # quaternion — penalize attitude error
        1e-2, 1e-2, 1e-2,    # angular velocity (rad/s)
    ])

    # Input weights — penalize fuel use / aggressive control
    R = np.diag([
        5e-1, 5e-1, 5e-1,    # force (N) — low to allow control authority
        1e-6, 1e-6, 1e-6,    # torque (N·m) — moderate
    ])

    # Terminal cost — heavy on final state
    QN = np.diag([
        1e2, 1e2, 1e2,       # position (z especially — want to land at altitude 0)
        1e4, 1e4, 1e4,       # velocity (want zero at touchdown)
        1e1, 1e1, 1e1, 1e1,  # quaternion (level attitude)
        1e1, 1e1, 1e1,       # angular velocity (no spinning)
    ])

    return Q,R,QN

def get_full_state_costs():
    Q = np.diag([
        1e-3, 1e-3, 1e-3,    # position (m) — moderate
        1e-0, 1e-0, 1e-0,    # velocity (m/s) — higher (you want soft touchdown)

        1e-1, 1e-1, 1e-1, 1e-1,  # quaternion — penalize attitude error
        1e-2, 1e-2, 1e-2,    # angular velocity (rad/s)
    ])

    # Input weights — penalize fuel use / aggressive control
    R = np.diag([
        5e-1, 5e-1, 5e-1,    # force (N) — low to allow control authority
        1e-6, 1e-6, 1e-6,    # torque (N·m) — moderate
    ])

    # Terminal cost — heavy on final state
    QN = np.diag([
        1e2, 1e2, 1e2,       # position (z especially — want to land at altitude 0)
        1e4, 1e4, 1e4,       # velocity (want zero at touchdown)
        1e1, 1e1, 1e1, 1e1,  # quaternion (level attitude)
        1e1, 1e1, 1e1,       # angular velocity (no spinning)
    ])

    return Q,R,QN

def get_initial_rv_state(altitiude_m: float = 20e3, downrange_angle_deg: float = 3):
    # 3 uprange in the -Y direction
    downrange_angle = downrange_angle_deg * DEG_TO_RAD
    r0_norm = R_MOON + altitiude_m # 20km altitude
    v0_norm = np.sqrt(GM_MOON / r0_norm) # circular 
    r0 = r0_norm * np.array([0, -np.sin(downrange_angle), np.cos(downrange_angle)])
    v0 = v0_norm * np.array([0, np.cos(downrange_angle), np.sin(downrange_angle)])

    s0 = np.array([
        *r0,
        *v0,
        1,0,0,0,0,0,0]) # doesn't matter for this because only r,v

    return s0,r0,v0

    
def remove_outliers(data, threshold_std=3, after_index = 0):
    result = data.copy()
    
    # Detect outliers
    median = np.median(data, axis=0, keepdims=True)
    mad = np.median(np.abs(data - median), axis=0, keepdims=True)  # Median Absolute Deviation  "The influence curve and its role in robust estimation"
    outlier_mask = np.abs(data - median) > threshold_std * mad
    
    # Replace outliers with previous value (or neighbor average)
    for i in np.where(outlier_mask)[0]:
        if i < after_index:
            continue
        if i == 0:
            # First point: use next value
            result[i] = result[i + 1]
        else:
            # Use previous value
            result[i] = result[i - 1]
    
    return result

if __name__ == "__main__":
    ### Define constants
    n = 13  # state dimension
    m = 6  # control dimension
    Q,R,QN = get_rv_costs()

    T = 200.0  # simulation time
    dt = 0.1  # sampling time

    mass_kg=50
    I = jnp.diag(jnp.array([8.0, 8.0, 5.0]))

    s0, r0, v0 = get_initial_rv_state(altitiude_m=20e3, downrange_angle_deg=3)
    print(f"Initial state: {s0}")


    s_goal = np.array([
        0,0,R_MOON, 
        0,0,0,
        1,0,0,0,0,0,0])
    

    ####################################################################################################
    #                                       For initial position iLQR
    ####################################################################################################
    
    # Initialize continuous-time and discretized dynamics
    # t: float, state: jnp.ndarray, disturbances: jnp.ndarray, mass_kg: float, I: np.ndarray):
    def next_state_wrapper(s, u):
        force_I = u[0:3]
        force_B = u[3:6]
        return lander_motion_inertial(s, force_I, force_B, dt, mass_kg, I) # mass and I from earlier


    print("Computing iLQR solution ... ", end="", flush=True)
    start = time.time()
    t = np.arange(0.0, T, dt)
    N = t.size - 1
    s_bar, u_bar, Y, y = ilqr_lander(next_state_wrapper, s0, s_goal, N, Q, R, QN, max_iters=20)
    print("done! ({:.2f} s)".format(time.time() - start), flush=True)

    ####################################################################################################
    #                                       Plotting results
    ####################################################################################################

    moon_offset = np.array([0, 0, R_MOON])


    print("Final position: ", s_bar[-1,0:3] - moon_offset)
    print("Final velocity: ", s_bar[-1,3:6])
    force_bar = u_bar[:,0:3]
    torque = u_bar[:,3:6]
    F_norms = np.linalg.norm(force_bar, axis=1)
    Tau_norms = np.linalg.norm(torque, axis=1)
    print("Max control force: ", np.max(F_norms))
    print("Max control torque: ", np.max(Tau_norms))


    # visualize_trajectory(s_bar, t, dt, offset = moon_offset, downsample_rate=5, moon_resolution = 35).show()

    # breakpoint()

    # plot_state_vector(t, s_bar[:,0:3] - np.tile(moon_offset, (t.size, 1)), s_bar[:,3:6], s_bar[:,10:13])
    # plot_control_effort(t[1:], u_bar[:,0:3], u_bar[:,3:6])

    # breakpoint()


    ####################################################################################################
    #                Turning position iLQR s_bar into a trajectory tracking for attitude iLQR
    ####################################################################################################

    # Angle with respect to vertical
    theta = [np.arctan2(np.sqrt(F[0]**2 + F[1]**2), F[2]) for F in force_bar] # angle between body z-axis and inertial z-axis
    q_rots = [angle_axis_to_q(theta_k, [1,0,0]) for theta_k in theta] # desired quaternion from angle

    # ---------------------------------------- Force ----------------------------------------
    u_bar_tracking = np.zeros_like(u_bar)
    u_bar_tracking[:,2] = F_norms # feedforward force from position iLQR, but no torque feedforward


    # ---------------------------------------- r,v,q ----------------------------------------
    s_bar_tracking = np.zeros_like(s_bar)
    s_bar_tracking[:,0:6] = s_bar[:,0:6] # track position and velocity from position iLQR
    s_bar_tracking[:-1,6:10] = q_rots # track attitude from angle of force vector
    s_bar_tracking[-1,6:10] = [1,0,0,0]

    # ---------------------------------------- w ----------------------------------------
    omega = np.zeros((N+1, 3))
    for i in range(N-1):
        omega[i,0] = -abs(theta[i+1] - theta[i])/dt # in -x direction (just for this trajectory) TODO change?

    omega = remove_outliers(omega, 4, 600)
    s_bar_tracking[:,10:13] = omega

    
    # ---------------------------------------- Torque ----------------------------------------
    alpha = np.diff(omega, axis=0) / dt
    alpha = remove_outliers(alpha, 3.5, 600) # smooth (manual lmao)

    torque = alpha @ I.T # feedforward torque from desired angular acceleration
    u_bar_tracking[:,3:6] = torque
    
    # lowkey kinda noisy sometimes so 
    plot_state_vector(t, s_bar_tracking[:,0:3] - np.tile(moon_offset, (t.size, 1)), s_bar_tracking[:,3:6], s_bar_tracking[:,10:13])
    breakpoint()

    visualize_trajectory(s_bar_tracking, t, dt, offset = moon_offset, downsample_rate=5, moon_resolution = 35).show()
    plot_control_effort(t[1:], u_bar_tracking[:,0:3], u_bar_tracking[:,3:6])

    breakpoint()

    ####################################################################################################
    #                With attitude iLQR from the start (full state cost)
    ####################################################################################################

    Q,R,QN = get_full_state_costs()

    # Initialize continuous-time and discretized dynamics
    # t: float, state: jnp.ndarray, disturbances: jnp.ndarray, mass_kg: float, I: np.ndarray):
    def next_state_wrapper(s, u):
        force_I = u[0:3]
        force_B = abs(u[5]) * np.array([0, 0, 1]) # just in z direction
        # force_B = u[3:6]
        return lander_motion(s, force_I, force_B, dt, mass_kg, I) # mass and I from earlier


    print("Computing iLQR solution ... ", end="", flush=True)
    start = time.time()
    t = np.arange(0.0, T, dt)
    N = t.size - 1
    s_bar, u_bar, Y, y = ilqr_lander(next_state_wrapper, s0, s_goal, N, Q, R, QN, max_iters=20,
                                     s_bar=s_bar_tracking, u_bar=u_bar_tracking)
    print("done! ({:.2f} s)".format(time.time() - start), flush=True)