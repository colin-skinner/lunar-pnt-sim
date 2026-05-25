import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from .sim.math.quaternion import quat_apply, norm, unit
from .constants import RAD_TO_DEG

# plt.style.use('dark_background')

########################################
#          Vector Helper               #
########################################

def plot_vec_3d(ax: Axes3D, p1: np.array, p2: np.array, color: str = None):
    xs = [p1[0], p2[0]]
    ys = [p1[1], p2[1]]
    zs = [p1[2], p2[2]]

    # print(f"{xs} {ys} {zs}")
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Panel
    if color is None:
        ax.plot(xs, ys, zs)
    else:
        ax.plot(xs, ys, zs, color=color)

########################################
#           1, 2, 3 columns            #
########################################

def plot_4(t: np.ndarray, thing, title=None, factor = 1, fmt = ''):
    f = plt.figure()
    thing = np.array(thing)

    plt.plot(t, thing[:,0] * factor, fmt, label = "W")
    plt.plot(t, thing[:,1] * factor, fmt, label = "X")
    plt.plot(t, thing[:,2] * factor, fmt, label = "Y")
    plt.plot(t, thing[:,3] * factor, fmt, label = "Z")
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    f.show()

def plot_3(t: np.ndarray, thing, title=None, factor = 1, fmt = ''):
    f = plt.figure()
    thing = np.array(thing)

    plt.plot(t, thing[:,0] * factor, fmt, label = "X")
    plt.plot(t, thing[:,1] * factor, fmt, label = "Y")
    plt.plot(t, thing[:,2] * factor, fmt, label = "Z")
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    f.show()

def plot_2(t, thing, title=None, factor = 1):
    f = plt.figure()
    plt.plot(t, thing * factor, label = "X")
    plt.plot(t, thing * factor, label = "Y")
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    f.show()

def plot_1(t, thing, title=None, factor = 1):
    f = plt.figure()
    plt.plot(t, thing * factor)
    if title:
        plt.title(title)
    plt.grid(True)
    f.show()

########################################
#            State Vectors             #
########################################

def plot_subplot(fig: Axes3D, t, data, datatype, axis, length_unit = "m", time_unit = "s"):
    fig.plot(t, data)
    fig.set_title(f"{axis} {datatype} vs. Time")
    fig.grid("True")
    fig.set_ylabel(f"{axis} ({length_unit})")
    fig.set_xlabel(f"Time ({time_unit})")

def plot_state_vector(
    t: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    figsize=(20, 9.5),
    show = True
):

    figure, axs = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    figure.suptitle("Position and Velocity)", fontsize=20)

    # Position
    plot_subplot(axs[0,0], t, r[:,0], "Position", "X", "m")
    plot_subplot(axs[0,1], t, r[:,1], "Position", "Y", "m")
    plot_subplot(axs[0,2], t, r[:,2], "Position", "Z", "m")

    # Velocity
    plot_subplot(axs[1,0], t, v[:,0], "Velocity", "X", "m/s")
    plot_subplot(axs[1,1], t, v[:,1], "Velocity", "Y", "m/s")
    plot_subplot(axs[1,2], t, v[:,2], "Velocity", "Z", "m/s")

    plt.show(block=False)

    # Angular velocity
    figure, axs = plt.subplots(nrows=1, ncols=3, figsize=(figsize[0], figsize[1] / 2))
    figure.suptitle("Angular Velocity", fontsize=20)

    plot_subplot(axs[0], t, w[:,0] * RAD_TO_DEG, "Angular Velocity", "X", "deg/s")
    plot_subplot(axs[1], t, w[:,1] * RAD_TO_DEG, "Angular Velocity", "Y", "deg/s")
    plot_subplot(axs[2], t, w[:,2] * RAD_TO_DEG, "Angular Velocity", "Z", "deg/s")
    
    plt.show(block=show)

def plot_control_effort(t: np.ndarray,
    force: np.ndarray,
    torque: np.ndarray,
    figsize=(20, 9.5),
    show = True
):
    """In body frame [N], [Nm]"""

    figure, axs = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    figure.suptitle("Drone Control Effort", fontsize=20)

    # Force
    plot_subplot(axs[0,0], t, force[:,0], "Force", "X", "N")
    plot_subplot(axs[0,1], t, force[:,1], "Force", "Y", "N")
    plot_subplot(axs[0,2], t, force[:,2], "Force", "Z", "N")

    # Torque
    plot_subplot(axs[1,0], t, torque[:,0], "Torque", "X", "N/m")
    plot_subplot(axs[1,1], t, torque[:,1], "Torque", "Y", "N/m")
    plot_subplot(axs[1,2], t, torque[:,2], "Torque", "Z", "N/m")

    plt.show(block=show)

########################################
#              Old 3D   (CHECK)                #
########################################

def plot_3d_helper(ax: Axes3D, states: np.ndarray, max_step = None):

    if max_step is None:
        max_step = -1

    p = states[:max_step, 0:3]
    v = states[:max_step, 3:6]

    # q = logger.actual_states[:max_step, 6:10]
    # w = logger.actual_states[:max_step, 10:13]




    r = states[:, 0:3]
    v = states[:, 3:6]

    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    vx = v[:, 0]
    vy = v[:, 1]
    vz = v[:, 2]
    

    # ax.set_xlim(-8,8)
    # ax.set_ylim(-8,8)
    # ax.set_zlim(0,5)

    ax.plot(x[:max_step],y[:max_step],z[:max_step])
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 3)

def debug_3d(states: np.ndarray,
             t: np.ndarray,
             dt: float,
             limits = None,
             title = 'Trajectory Debug',
             desired_data: dict[float, tuple[float, float]] = None,
             figsize = (20,10),
             length_unit = 'm',
             start_time_s = 0, interval = 1):

    max_step = len(states)

    plt.ion()

    fig = plt.figure(figsize = figsize)
    fig.suptitle(title, fontsize = 20)
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    

    start_step = int(start_time_s / dt) + 1

    for step in range(start_step, max_step, interval):

        p = states[:step, 0:3]
        curr_p = p[-1]
        q = states[:step, 6:10]
        curr_q_B2I = q[-1]

        # torque = logger.actual_torques[step]
        # q_d = logger.drone_desired_quat[step]
        # torque = logger.drone_commanded_torques[step]
        # p_d_err = logger.drone_p_d_error[step]

        # thrust = logger.drone_commanded_force[step]



        plot_3d_helper(ax, states, step)
        # ax.text2D(0.05, 0.95, f"Thrust: {norm(thrust)}", transform=ax.transAxes)

        
        # Coordinate system
        # plot_vec_3d(ax, curr_p, curr_p + unit(torque), 'black')
        plot_vec_3d(ax, curr_p, curr_p + quat_apply(curr_q_B2I, [5,0,0]), 'red')
        plot_vec_3d(ax, curr_p, curr_p + quat_apply(curr_q_B2I, [0,5,0]), 'green')
        plot_vec_3d(ax, curr_p, curr_p + quat_apply(curr_q_B2I, [0,0,5]), 'blue')

        # plot_vec_3d(ax, curr_p, curr_p + unit(quat_apply(q_d, [0.5,0,0])), 'purple')
        # plot_vec_3d(ax, curr_p, curr_p + unit(quat_apply(q_d, [0,0.5,0])), 'orange')
        # plot_vec_3d(ax, curr_p, curr_p + unit(quat_apply(q_d, [0,0,0.5])), 'black')
        # plot_vec_3d(ax, curr_p, curr_p + unit(torque), 'red')


        # plot_vec_3d(ax, curr_p, curr_p + p_d_err, 'brown')
        # plot_vec_3d(ax, curr_p, curr_p + thrust, 'gray')
        # fig.show()

        # ax.set_xlim(min(logger.actual_states[:max_step, 0] - 0.5), max(logger.actual_states[:max_step, 0]) + 0.5)
        # ax.set_ylim(min(logger.actual_states[:max_step, 1] - 0.5), max(logger.actual_states[:max_step, 1]) + 0.5)
        # ax.set_zlim(min(logger.actual_states[:max_step, 2] - 0.5), max(logger.actual_states[:max_step, 2]) + 0.5)
        if limits is not None:
            ax.set_xlim(*limits[0])
            ax.set_ylim(*limits[1])
            ax.set_zlim(*limits[2])

        ax.set_title(f"{title}: {t[step]:.2f}s")

        ax.legend(["Trajectory", "Desired Trajectory", "x_axis", "y_axis", "z_axis", "x_d", "y_d", "z_d", "p_err"])
        # ax.set_aspect('equal', adjustable='box')
        # ax.axis('square')
        plt.pause(dt/10)

        # if step + interval + 1 > max_step:
        #     if input("Press w to watch again:") == "w":
        #         debug_3d(logger, title, desired_data, figsize, length_unit, start_time_s, interval)


        ax.cla()

########################################
#            New Vis                   #
########################################