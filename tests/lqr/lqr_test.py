import time

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np

from scipy.integrate import odeint

import sys
sys.path.append("../..")

from lunanav.sim.quaternion import angle_axis_to_q, unit, quat_apply, conj
from lunanav.plotting import plot_state_vector, debug_3d, plot_control_effort, plot_4
from lunanav.constants import GM_MOON, R_MOON

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

    s_bar = np.zeros((N + 1, n))
    s_bar[0] = s0
    for k in range(N):
        s_bar[k + 1] = f(s_bar[k], u_bar[k])

        x = s_bar[k + 1]
        if k % 50 == 0:
            print(f"Step {k}: pos={x[0:3]}, vel={x[3:6]}, q={x[6:10]}, angvel={x[10:13]}")
        
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e10):
            print(f"Diverged at step {k}")
            break

    ds = np.zeros((N + 1, n))
    du = np.zeros((N, m))

    # iLQR loop
    converged = False
    for i in range(max_iters):

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
            H_uu = R + B[k].T @ P[k+1] @ B[k]

            q_k = Q @ (s_bar[k] - s_goal) # From part (b)
            r_k = R @ u_bar[k] # From part (b)

            h_x = q_k + A[k].T @ p[k+1]
            h_u = r_k + B[k].T @ p[k+1]

            

            try:
                Y[k] = -np.linalg.pinv(H_uu) @ H_xu.T
            except np.linalg.LinAlgError:
                # print(B[4])
                # print(B[k-1])
                print(B[k])
                # print(P[k])
                print(P[k+1])
                print(k)
                breakpoint()
            y[k] = -np.linalg.pinv(H_uu) @ h_u
            
            P[k] = H_xx + H_xu @ Y[k]
            p[k] = h_x + H_xu @ y[k]



        # Forward pass
        for k in range(N):
            du[k] = y[k] + Y[k] @ ds[k]
            ds[k+1] = f(s_bar[k] + ds[k], u_bar[k] + du[k]) - s_bar[k+1]


        s_bar = s_bar + ds
        u_bar = u_bar + du
        
        #######################################################################

        if i % 10 == 0:
            print(f"{i}: {np.max(np.abs(du))}")

        if np.max(np.abs(du)) < eps:
            converged = True
            break
    # if not converged:
    #     raise RuntimeError("iLQR did not converge!")
    return s_bar, u_bar, Y, y

def get_costs():
    # State weights — tune by physical units
    Q = np.diag([
        1e-3, 1e-3, 1e-3,    # position (m) — moderate
        1e-2, 1e-2, 1e-2,    # velocity (m/s) — higher (you want soft touchdown)
        1e-1, 1e-1, 1e-1, 1e-1,  # quaternion — penalize attitude error
        1e-2, 1e-2, 1e-2,    # angular velocity (rad/s)
    ])

    # Input weights — penalize fuel use / aggressive control
    R = np.diag([
        1e-2, 1e-2, 1e-4,    # force (N) — low to allow control authority
        1e-2, 1e-2, 1e-2,    # torque (N·m) — moderate
    ])

    # Terminal cost — heavy on final state
    QN = np.diag([
        1e2, 1e2, 1e3,       # position (z especially — want to land at altitude 0)
        1e3, 1e3, 1e1,       # velocity (want zero at touchdown)
        1e1, 1e1, 1e1, 1e1,  # quaternion (level attitude)
        1e1, 1e1, 1e1,       # angular velocity (no spinning)
    ])

    return Q,R,QN

if __name__ == "__main__":
    # Define constants
    n = 13  # state dimension
    m = 6  # control dimension
    Q,R,QN = get_costs()
    T = 100.0  # simulation time
    dt = 0.1  # sampling time

    # Spacecraft
    mass = 0.5 # kg
    I = 5 * np.eye(3)
    I_inv = jnp.linalg.pinv(I)


    r = R_MOON + 15 # GEO
    v = np.sqrt(GM_MOON/r) # circular velocity

    # Boundary conditions
    s0 = np.array([
    1000,2000, r, 
    0 ,1000, 0,
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
    # input: force_inertial, torque_body
    # t: float, state: jnp.ndarray, disturbances: jnp.ndarray, mass_kg: float, I: np.ndarray):
    @jax.jit
    def deriv_wrapper(s, u):
        q_B2I = unit(s[6:10])
        u_better = jnp.concatenate((quat_apply(q_B2I, u[0:3]), u[3:6]))
        return rigid_body_derivative(0, s, u_better, mass, I, I_inv)
        
    f = jax.jit(deriv_wrapper)
    fd = jax.jit(lambda s, u, dt=dt: s + dt * f(s, u))

    print("Computing iLQR solution ... ", end="", flush=True)
    start = time.time()
    t = np.arange(0.0, T, dt)
    N = t.size - 1
    s_bar, u_bar, Y, y = ilqr(fd, s0, s_goal, N, Q, R, QN, max_iters=100)
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

    plot_state_vector(t, s)
    plt.show(block=False)
    plot_4(t, s[:,6:10])
    plt.show(block=False)
    plot_control_effort(t, u)
    plt.show()

    x = s[:,0]
    y = s[:,1]
    z = s[:,2]

    limits = np.array([
        [min(x), max(x)],
        [min(y), max(y)],
        [min(z), max(z)]
    ])
    

    debug_3d(s, t, dt, limits=limits)
