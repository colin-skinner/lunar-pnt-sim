import numpy as np
from tqdm import tqdm
import jax
from ..sim.dynamics import linearize

# Make jacobian
# LQR
# Linearize, forward pass, backward pass

def ilqr_lander(f,
         state0: np.ndarray,
         s_goal: np.ndarray,
         N: int,
         Q: np.ndarray,
         R: np.ndarray,
         QN: np.ndarray,
         s_bar: np.ndarray = None,
         u_bar: np.ndarray = None,
         eps: float = 1e-3,
         max_iters: int = 1000):
    """Compute the iLQR set-point tracking solution.

    Arguments
    ---------
    f : callable
        A function describing the discrete-time dynamics, such that
        `s[k+1] = f(s[k], u[k])`.
    state0 : numpy.ndarray
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
    u_bar = np.zeros((N, m)) if u_bar is None else u_bar
    s_bar = np.zeros((N + 1, n)) if s_bar is None else s_bar
    s_bar[0] = state0
    for k in range(N):
        s_bar[k + 1] = f(s_bar[k], u_bar[k])
    ds = np.zeros((N + 1, n))
    du = np.zeros((N, m))

    # iLQR loop
    converged = False
    bar = tqdm(range(max_iters), desc="iLQR")
    for i in bar:

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

        for k in tqdm(range(N-1, -1, -1)):
            H_xu = A[k].T @ P[k+1] @ B[k]
            H_xx = Q + A[k].T @ P[k+1] @ A[k]
            H_uu = R + B[k].T @ P[k+1] @ B[k]



        # Check for problems
            # if np.any(np.isnan(H_uu)) or np.any(np.isinf(H_uu)):
            #     print(f"Step {k}: NaN/Inf in H_uu")
            #     print(f"  B[k] rank: {np.linalg.matrix_rank(B[k])}")
            #     print(f"  P[k+1] condition: {np.linalg.cond(P[k+1])}")
            #     break
            
            # det = np.linalg.det(H_uu)
            # if abs(det) < 1e-15:
            #     print(f"Step {k}: H_uu is singular (det={det:.2e})")
            #     print(f"  H_uu eigenvalues: {np.linalg.eigvals(H_uu)}")
            #     print(f"  R:\n{R}")
            #     break

            
            q_k = Q @ (s_bar[k] - s_goal) # From part (b)
            r_k = R @ u_bar[k] # From part (b)

            h_x = q_k + A[k].T @ p[k+1]
            h_u = r_k + B[k].T @ p[k+1]

            Y[k] = -np.linalg.pinv(H_uu) @ H_xu.T
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

        bar.set_postfix(du_err=f"{np.max(np.abs(du)):.2e}", refresh=True)

        if np.max(np.abs(du)) < eps:
            # converged = True
            print(f"DONE?!?!?! du max abs: {np.max(np.abs(du))}")
            break
    # if not converged:
    #     raise RuntimeError("iLQR did not converge!")
    return s_bar, u_bar, Y, y