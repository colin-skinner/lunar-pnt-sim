import time

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np

from scipy.integrate import odeint

from debug import debug_trajectory

import sys
sys.path.append("../..")

from lunanav.sim.quaternion import angle_axis_to_q, unit, quat_apply, conj
from lunanav.plotting import plot_state_vector, debug_3d, plot_control_effort, plot_4, plot_3
from lunanav.constants import GM_MOON, R_MOON
from lunanav.sim.simulator import lander_motion, linearized_lander_motion
from lunanav.visualization import visualize_trajectory

moon_offset =  [0,0,R_MOON,0,0,0,0,0,0,0,0,0,0]


def linearize(f, s, u):
    """Linearize the function `f(s, u)` around `(s, u)`.

    Arguments
    ---------
    f : callable
        A nonlinear function with call signature `f(s, u)`.
    s : numpy.ndarray
        The state (1-D).
    u : numpy.ndarray
        The control input (1-D).

    Returns
    -------
    A : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `s`.
    B : numpy.ndarray
        The Jacobian of `f` at `(s, u)`, with respect to `u`.
    """
    # WRITE YOUR CODE BELOW ###################################################
    # INSTRUCTIONS: Use JAX to compute `A` and `B` in one line.
    A,B = jax.jacobian(f, argnums=(0,1))(s, u)
    ###########################################################################
    return A, B

def ilqr(f, s0, s_goal, N, Q, R, QN, eps=1e-3, max_iters=1000):
    """Compute the iLQR set-point tracking solution.

    Arguments
    ---------
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    s0 : numpy.ndarray
        The initial state (1-D).
    s_goal : numpy.ndarray
        The goal state (1-D).
    N : int
        The time horizon of the LQR cost function.
    Q : numpy.ndarray
        The state cost matrix (2-D).
    R : numpy.ndarray
        The control cost matrix (2-D).
    QN : numpy.ndarray
        The terminal state cost matrix (2-D).
    eps : float, optional
        Termination threshold for iLQR.
    max_iters : int, optional
        Maximum number of iLQR iterations.

    Returns
    -------
    s_bar : numpy.ndarray
        A 2-D array where `s_bar[k]` is the nominal state at time step `k`,
        for `k = 0, 1, ..., N-1`
    u_bar : numpy.ndarray
        A 2-D array where `u_bar[k]` is the nominal control at time step `k`,
        for `k = 0, 1, ..., N-1`
    Y : numpy.ndarray
        A 3-D array where `Y[k]` is the matrix gain term of the iLQR control
        law at time step `k`, for `k = 0, 1, ..., N-1`
    y : numpy.ndarray
        A 2-D array where `y[k]` is the offset term of the iLQR control law
        at time step `k`, for `k = 0, 1, ..., N-1`
    """
    if max_iters <= 1:
        raise ValueError("Argument `max_iters` must be at least 1.")
    n = Q.shape[0]  # state dimension
    m = R.shape[0]  # control dimension

    # Initialize gains `Y` and offsets `y` for the policy
    Y = np.zeros((N, m, n))
    y = np.zeros((N, m))

    # Initialize the nominal trajectory `(s_bar, u_bar`), and the
    # deviations `(ds, du)`
    u_bar = np.zeros((N, m))

    best_s_bar = None
    best_u_bar = None
    best_error = np.inf

    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k + 1] = f(s_bar[k], u_bar[k])

        x = s_bar[k + 1]
        if k % 50 == 0:
            print(f"Step {k}: pos={x[0:3]}, vel={x[3:6]}, q={x[6:10]}, angvel={x[10:13]}")
        
        if np.any(np.isnan(x)):
            print(f"{np.any(np.isnan(x))=}")
            plt.plot(s_bar[k-1,:2])
            plt.show(block=False)
            raise ValueError(f"Diverged at step {k}")

    ds = np.zeros((N + 1, n))
    du = np.zeros((N, m))

    # iLQR loop
    converged = False
    for i in range(max_iters):
        
        # if i % 10 == 0 or i < 5:
        print(f"===================== BEFORE iteration {i} =====================")
        debug_trajectory(s_bar, u_bar, s_goal, i, N, dt, mass=100)

        # PART (c) ############################################################

        # Linearize the dynamics at each step `k` of `(s_bar, u_bar)`
        A, B = jax.vmap(linearize, in_axes=(None, 0, 0))(f, s_bar[:-1], u_bar)
        A, B = np.array(A), np.array(B)

        ds = np.zeros_like(ds)
        du = np.zeros_like(du)
        
        # 1. Backward pass
        P = np.zeros((N + 1, n, n))
        p = np.zeros((N + 1, n))
        
        q_N = QN @ (s_bar[N] - s_goal) # From part (b)
        
        P[N] = QN
        p[N] = q_N

        for k in range(N-1, -1, -1):
            
            
            H_xu = A[k].T @ P[k+1] @ B[k]
            H_xx = Q + A[k].T @ P[k+1] @ A[k]
            H_uu = R + B[k].T @ P[k+1] @ B[k]
            # print(B[k])
            # print(P[k+1])
            # breakpoint

            q_k = Q @ (s_bar[k] - s_goal) # From part (b)
            r_k = R @ u_bar[k] # From part (b)

            h_x = q_k + A[k].T @ p[k+1]
            h_u = r_k + B[k].T @ p[k+1]

            # STRONGER regularization
            regularization = 1e-2  # Increased from 1e-4
            H_uu_reg = H_uu + regularization * np.eye(m)
            
            # Check condition number before solving
            cond = np.linalg.cond(H_uu_reg)
            if cond > 1e10:
                print(f"  Warning: H_uu ill-conditioned at k={k}, cond={cond:.2e}")
                regularization = 1.0  # Much stronger
                H_uu_reg = H_uu + regularization * np.eye(m)
            
            try:
                Y[k] = -np.linalg.solve(H_uu_reg, H_xu.T)
                y[k] = -np.linalg.solve(H_uu_reg, h_u)
            except np.linalg.LinAlgError:
                print(f"  Using pseudoinverse at k={k}")
                Y[k] = -np.linalg.pinv(H_uu_reg) @ H_xu.T
                y[k] = -np.linalg.pinv(H_uu_reg) @ h_u
                
                # Check for NaN in gains
                if np.any(np.isnan(Y[k])) or np.any(np.isnan(y[k])):
                    print(f"  NaN in gains at k={k}, stopping iteration")
                    converged = True  # Stop but keep previous good solution
                    break

            # print(f"k={k:3d} | |A|={np.linalg.norm(A[k]):.2e}" + 
            #     f" |B|={np.linalg.norm(B[k]):.2e}" + 
            #     f" |P[k+1]|={np.linalg.norm(P[k+1]):.2e}" + 
            #     f" |H_uu|={np.linalg.norm(H_uu):.2e}" + 
            #     # f" cond(H_uu)={np.linalg.cond(H_uu):.2e}" + 
            #     # f" eig_min(H_uu)={np.min(np.linalg.eigvalsh(H_uu)):.2e}" + 
            #     f" |p[k+1]|={np.linalg.norm(p[k+1]):.2e}" + 
            #     f" |s_bar|={np.linalg.norm(s_bar[k]):.2e}" + 
            #     f" |u_bar|={np.linalg.norm(u_bar[k]):.2e}")
            # if any(np.isnan(s_bar[k])):
            #     breakpoint()


            # try:
            #     Y[k] = -np.linalg.pinv(H_uu) @ H_xu.T
            # except np.linalg.LinAlgError:
            #     print(s_bar[i])
            #     # print(B[4])
            #     # print(B[k-1])
            #     print(B[k])
            #     # print(P[k])
            #     print(P[k+2])
            #     print(k)
            #     breakpoint()

            # y[k] = -np.linalg.pinv(H_uu) @ h_u

            # try:
            #     Y[k] = -np.linalg.solve(H_uu_reg, H_xu.T)
            #     y[k] = -np.linalg.solve(H_uu_reg, h_u)
            #     P[k] = H_xx + H_xu @ Y[k]
            #     p[k] = h_x + H_xu @ y[k]
            # except np.linalg.LinAlgError:
            #     print(f"Linear algebra error at k={k}")
            #     Y[k] = -np.linalg.pinv(H_uu_reg) @ H_xu.T
            #     y[k] = -np.linalg.pinv(H_uu_reg) @ h_u
            #     P[k] = H_xx + H_xu @ Y[k]
            #     p[k] = h_x + H_xu @ y[k]

        # After backward pass, before forward pass
        # if i % 10 == 0:

        # if any(np.isnan(s_bar[-1])):
        #     print(s_bar[-1])
        # visualize_trajectory(s_bar - np.tile(moon_offset, (N+1, 1)), None, dt, title="Lunar Descent Trajectory with LOS Vectors", show_lander=True, show_moon=True).show()
            # breakpoint()
        # print("\n" + "="*60)
        # print(f"Iteration {i}")
        # print(f"Initial state: {s_bar[0, 0:3]}")
        # print(f"Final state:   {s_bar[-1, 0:3]}")
        # print(f"Goal state:    {s_goal[0:3]}")
        # print(f"Position error: {np.linalg.norm(s_bar[-1, 0:3] - s_goal[0:3]):.2f} m")
        # print(f"Velocity error: {np.linalg.norm(s_bar[-1, 3:6] - s_goal[3:6]):.2f} m/s")
        # print("="*60 + "\n")
        # visualize_trajectory(s_bar - np.tile(moon_offset, (N+1, 1)), None, dt, title="Lunar Descent Trajectory with LOS Vectors", show_lander=True, show_moon=True).show()
        # breakpoint()

        if converged:
            print("Stopping due to numerical issues, using previous solution")
            break

        print(f"===================== AFTER BACKWARD {i} =====================")
        debug_trajectory(s_bar, u_bar, s_goal, i, N, dt, mass=100)

        # Forward pass
        # for k in range(N):
        #     du[k] = y[k] + Y[k] @ ds[k]
        #     ds[k+1] = f(s_bar[k] + ds[k], u_bar[k] + du[k]) - s_bar[k+1]
        # Forward pass
        ds[0] = np.zeros(n)  # Initial state deviation is zero (we start at s0)

        for k in range(N):
            # Compute control update using current state deviation
            du[k] = y[k] + Y[k] @ ds[k]
            
            # Simulate forward with updated control
            s_new = s_bar[k] + ds[k]
            u_new = u_bar[k] + du[k]
            s_next = f(s_new, u_new)
            
            # State deviation at next step
            ds[k+1] = s_next - s_bar[k+1]

            # CHECK IF UNDERGROUND
            # altitude = (s_bar[k+1, 2] + ds[k+1, 2]) - R_MOON
            # if altitude < 0:
            #     print(f"  WARNING: Went underground at step {k}, alt={altitude:.1f}m")
            #     # Truncate trajectory here
            #     N_actual = k + 1
            #     # s_bar = s_bar[:N_actual+1]
            #     # u_bar = u_bar[:N_actual]
            #     break


        s_bar = s_bar + ds
        u_bar = u_bar + du

        pos_err = np.linalg.norm(s_bar[-1, 0:3] - s_goal[0:3])
        vel_err = np.linalg.norm(s_bar[-1, 3:6] - s_goal[3:6])
        
        if pos_err < 1.0 and vel_err < 1.0:  # Within 1m and 1m/s
            print(f"\nGood enough solution at iteration {i}:")
            print(f"  Position error: {pos_err:.2f} m")
            print(f"  Velocity error: {vel_err:.2f} m/s")
            converged = True
            break

        print(f"===================== AFTER FORWARD {i} =====================")
        debug_trajectory(s_bar, u_bar, s_goal, i, N, dt, mass=100)
        # breakpoint()

        #######################################################################
        # breakpoint()
        # if i % 10 == 0:
        print(f"{i}: err_du={np.max(np.abs(du))}, err_ds={np.max(np.abs(ds))}")
            # moon_offset_vec = np.tile(moon_offset, (N+1, 1))
            # visualize_trajectory(s_bar, None, dt, title="Lunar Descent Trajectory with LOS Vectors", show_lander=True, show_moon=False).show()
            # plt.figure()
            # plt.plot(u_bar[:,0])
            # plt.show(block=False)
            # plt.figure()
            # plt.plot(u_bar[:,1:])
            # plt.show(block=False)
            # breakpoint()


        
        # Compute errors
        pos_err = np.linalg.norm(s_bar[-1, 0:3] - s_goal[0:3])
        vel_err = np.linalg.norm(s_bar[-1, 3:6] - s_goal[3:6])
        total_err = pos_err + vel_err
        
        # Save best solution
        if total_err < best_error:
            best_error = total_err
            best_s_bar = s_bar.copy()
            best_u_bar = u_bar.copy()
        
        print(f"Iter {i}: pos_err={pos_err:.2f}m, vel_err={vel_err:.2f}m/s")
        
        # Check for convergence
        if pos_err < 1 and vel_err < 1:
            print(f"\n✓ Converged at iteration {i}!")
            print(f"  Final position error: {pos_err:.3f} m")
            print(f"  Final velocity error: {vel_err:.3f} m/s")
            converged = True
            break

        if np.max(np.abs(du)) < eps:
            converged = True
            break
    # if not converged:
    #     raise RuntimeError("iLQR did not converge!")
    return best_s_bar, best_u_bar, Y, y

def get_costs():
    """Properly scaled costs for lunar landing"""
    Q = np.diag([
        1e-6, 1e-6, 1e-6,      # position during flight
        1e-4, 1e-4, 1e-4,      # velocity during flight
        1e-6, 1e-6, 1e-6, 1e-6,  # quaternion
        1e-6, 1e-6, 1e-6,      # angular velocity
    ])

    R = np.diag([
        1e-6, 1e-6, 1e-6,      # thrust - allow moderate control
        1e-4, 1e-4, 1e-4,      # torque
    ])

    QN = np.diag([
        1e2, 1e2, 1e3,         # final position - MUST be at target
        1e3, 1e3, 1e3,         # final velocity - MUST be zero
        1e0, 1e0, 1e0, 1e0,    # final attitude
        1e0, 1e0, 1e0,         # final angular rate
    ])

    return Q, R, QN

if __name__ == "__main__":
    # Define constants
    n = 13  # state dimension
    m = 6  # control dimension (Thruster force and ACS torques)
    Q,R,QN = get_costs()
    T = 100.0  # simulation time
    dt = 1  # sampling time

    # Spacecraft
    mass = 100 # kg
    I = np.eye(3)
    I_inv = jnp.linalg.pinv(I)


    r = R_MOON + 10000 # GEO
    v = np.sqrt(GM_MOON/r) # circular velocity

    # Boundary conditions
    s0 = np.array([
    0,-5000, r, 
    0 ,100, -50,
    1,0,0,0,
    # *angle_axis_to_q(90, [1,0,0], True),
    0,0,0])

    print(f"Initial state: {s0}")


    s_goal = np.array([
        0,0, R_MOON, 
        0 ,0, 0,
        1,0,0,0,
        0,0,0])
    
    # Initialize continuous-time and discretized dynamics 
    # input: force_body, torque_body
    # t: float, state: jnp.ndarray, disturbances: jnp.ndarray, mass_kg: float, I: np.ndarray):
    def deriv_wrapper(s, u):
        return lander_motion(s, u[0:3], u[3:6], dt, mass, I)

    f = jax.jit(deriv_wrapper)
    # fd = jax.jit(lambda s, u, dt=dt: s + dt * f(s, u))

    print("Computing iLQR solution ... ", end="", flush=True)
    start = time.time()
    t = np.arange(0.0, T, dt)
    N = t.size - 1
    s_bar, u_bar, Y, y = ilqr(f, s0, s_goal, N, Q, R, QN, max_iters=100)
    print("done! ({:.2f} s)".format(time.time() - start), flush=True)


    # Simulate on the true continuous-time system
    print("Simulating ... ", end="", flush=True)
    start = time.time()
    s = np.zeros((N + 1, n))
    u = np.zeros((N, m))
    s[0] = s0
    for k in range(N):
        # PART (d) ################################################################
        # INSTRUCTIONS: Compute either the closed-loop or open-loop value of
        # `u[k]`, depending on the Boolean flag `closed_loop`.
        u[k] = u_bar[k] + Y[k]@(s[k] - s_bar[k]) + y[k]
            
        ###########################################################################
        s[k + 1] = odeint(lambda s, t: f(s, u[k]), s[k], t[k : k + 2])[1]
    print("done! ({:.2f} s)".format(time.time() - start), flush=True)
    visualize_trajectory(s_bar - np.tile(moon_offset, (N+1, 1)), None, dt, title="Lunar Descent Trajectory with LOS Vectors", show_lander=True, show_moon=True).show()
    plot_3(t, s[:,0:3])
    plot_3(t[:-1], u[:,0:3])
    plot_3(t[:-1], u[:,3:6])
    breakpoint()
    plot_state_vector(t, s[:,0:3], s[:,3:6], s[:,6:10], s[:,10:13])
    # plt.show(block=False)
    # plot_4(t, s[:,6:10])
    # plt.show(block=False)
    # plot_control_effort(t, u)
    # plt.show()

    x = s[:,0]
    y = s[:,1]
    z = s[:,2]

    limits = np.array([
        [min(x), max(x)],
        [min(y), max(y)],
        [min(z), max(z)]
    ])
    

    debug_3d(s, t, dt, limits=limits)
