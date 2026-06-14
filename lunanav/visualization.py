import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.linalg import svd
from pathlib import Path
import jax
import jax.numpy as jnp
from .sim.quaternion import quat_apply, unitize_state
from .constants import R_MOON
from .sim.sensors import SensorSuite
from .estimation.ekf import ekf_predict_state_only

# Define specific colors
x_axis_color = 'red'
y_axis_color = 'green'
z_axis_color = 'blue'

# def moon_surface(xx, yy, zz):
#     """Returns go.Surface of Moon"""
#     return go.Surface(x=xx, y=yy, z=zz, colorscale=[[0, '#333333'], [1, '#555555']],
#                              showscale=False, name='Moon', hoverinfo='skip')

def moon_surface(radius=R_MOON, offset: np.ndarray = np.zeros(3), resolution=25, color='#444444'):
    """Returns go.Surface of Moon as a sphere. Offset is how far UP the coordinate system is shifted"""
    # Create sphere using spherical coordinates
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v)) - offset[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) - offset[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) - offset[2]
    
    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, '#333333'], [1, '#555555']],
        showscale=False,
        name='Moon',
        hoverinfo='skip',
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.2, roughness=0.8),
        lightposition=dict(x=100000, y=100000, z=100000)
    )

def add_lander(position, show_lander=True):
    """Add lander marker to the figure."""
    if not show_lander:
        return []
    return go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]],
                                 mode='markers', marker=dict(size=6, color='black'),
                                 name='Lander', hoverinfo='skip')

def add_gradient_trajectory(r, t, colorscale='Viridis', downsample_rate=5):
    """Return go.Scatter3d of trajectory colored by time."""
    return go.Scatter3d(
        x=r[::downsample_rate, 0], y=r[::downsample_rate, 1], z=r[::downsample_rate, 2],
        mode='lines',
        line=dict(color=t, colorscale=colorscale, width=3, showscale=False),
        name='Trajectory'  # Name for trajectory
    )

def add_solid_trajectory(r, color='red', downsample_rate=5):
    """Return go.Scatter3d of trajectory colored by time."""
    return go.Scatter3d(
        x=r[::downsample_rate, 0], y=r[::downsample_rate, 1], z=r[::downsample_rate, 2],
        mode='lines',
        line=dict(color=color, width=2, showscale=False),
        name='Solid Trajectory'  # Name for solid trajectory
    )


def draw_vectors(r, q, names, vectors, colors, scale=1.0):
    """Draw vectors from a specified position according to the given orientation (quaternion)."""
    traces = []
    for name, vec, color in zip(names, vectors, colors):
        # Transform the vector using the orientation defined by the quaternion
        transformed_vector = quat_apply(q, vec) * scale  # Scale the vector length
        
        # Create the line trace for the vector
        traces.append(go.Scatter3d(
            x=[r[0], r[0] + transformed_vector[0]], 
            y=[r[1], r[1] + transformed_vector[1]], 
            z=[r[2], r[2] + transformed_vector[2]], 
            mode='lines',
            line=dict(color=color, width=3),  # You can adjust the width and color
            name=name,hoverinfo='skip'
        ))
    return traces

def get_body_axes(r, q, axis_size, show_body_axes=True):
    """Returns go.Scatter3d traces for body axes based on current position and orientation."""
    if not show_body_axes:
        return []
    return draw_vectors(
        r,q,
        names=['X-axis', 'Y-axis', 'Z-axis'],
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        colors=[x_axis_color, y_axis_color, z_axis_color],
        scale=axis_size
    )


def visualize_trajectory(
    trajectories: np.ndarray | list,
    t: np.ndarray = None,
    dt: float = 0.1,
    axis_scale: float = 1000.0,
    offset = None,
    *, 
    title: str = "Lunar Descent Trajectories",
    other_vecs: dict = None,
    show_body_axes: bool = True,
    show_lander: bool = True,
    downsample_rate: int = 5,
    moon_resolution: int = 35
):
    """
    Simple interactive 3D trajectory visualizer for one or multiple trajectories.
    
    Args:
        trajectories: list of [N, 13] arrays (each containing r, v, q, omega for a trajectory)
        t: time array (auto-generated if None)
        dt: time step in seconds
        axis_scale: length of body axis vectors (meters)
        title: plot title
        
    Returns:
        plotly Figure
    """

    if offset is None:
        offset = np.array([0,0,R_MOON])

    if isinstance(trajectories, np.ndarray):
        trajectories = [trajectories]  # Wrap in list if single trajectory provided

    n = len(trajectories[0])
    offset_full_state = np.tile(offset, (n, 1))
    offset_full_state = np.pad(offset_full_state, ((0, 0), (0, 10)))
    trajectories = [t - offset_full_state for t in trajectories]
    
    fig = go.Figure()

    all_frames = []
    
    for index, states in enumerate(trajectories):
        n_steps = len(states)
        if t is None:
            t = np.arange(n_steps) * dt
        
        # Extract states
        r = states[:, 0:3]
        q = states[:, 6:10]
        
        # Moon surface
        xx, yy = np.meshgrid(np.linspace(-10000, 10000, 5), np.linspace(-10000, 10000, 5))
        zz = np.full_like(xx, r[0, 2])  # Use the initial altitude
        moon_surface_trace = moon_surface(radius=R_MOON, offset = offset, resolution=moon_resolution)
        fig.add_trace(moon_surface_trace)

        # Lander marker
        if show_lander:
            fig.add_trace(add_lander(r[0]))

        # Solid trajectory for visibility
        solid_traj = add_solid_trajectory(r, color=f'gold' if index % 2 == 0 else 'silver', downsample_rate=downsample_rate)  # Different colors for different trajectories
        fig.add_trace(solid_traj)

        # Body axes setup 
        traj_scale = np.max(np.linalg.norm(r - r[0], axis=1))  # max distance from start
        axis_size = min(axis_scale, traj_scale * 0.1)  # 10% of trajectory extent or user-specified
        fig.add_traces(get_body_axes(r[0], q[0], axis_size, show_body_axes=show_body_axes))

        # Other vectors (e.g., velocity, acceleration) if provided
        if other_vecs is not None:
            names, vecs, colors, vec_scale = other_vecs["names"], other_vecs["vecs"], other_vecs["colors"], other_vecs.get("scale", 1e3)
            fig.add_traces(draw_vectors(r[0], q[0], names, vecs, colors, scale=vec_scale))

        # Animation frames
        frames = []
        for step in range(0, n_steps, downsample_rate):
            pos = r[step]
            q_curr = q[step]
            frame_data = [moon_surface_trace]
            if show_lander:
                frame_data.append(add_lander(r[step], show_lander=show_lander))
            if other_vecs is not None:
                frame_data += draw_vectors(pos, q_curr, names, vecs, colors, scale=vec_scale)  # Add other vectors if they exist
            frame_data.append(solid_traj)  # Add solid trajectory
            frame_data += get_body_axes(pos, q_curr, axis_size, show_body_axes=show_body_axes)  # Include body axes for the current state
            frames.append(go.Frame(data=frame_data, name=str(step)))  # Append the frame data
        all_frames.extend(frames)  # Combine frames from all trajectories

    fig.frames = all_frames  # Set the frames for animation 
    
    # Slider and Updatemenus
    sliders = [{'active': 0, 'yanchor': 'top', 'y': 0, 'xanchor': 'left', 'x': 0.1, 'len': 0.9,
                'currentvalue': {'prefix': 'Time: ', 'suffix': ' s', 'visible': True},
                'steps': [{'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                           'method': 'animate', 'label': f'{t[i]:.1f}'} for i in range(n_steps)]
    }]

    # Adding Play/Pause buttons
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}],
                'label': 'Play', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                'label': 'Pause', 'method': 'animate'}
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': True,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )
    
    # Auto-scale to the maximum trajectory extent
    all_r = np.concatenate([states[:, 0:3] for states in trajectories])  # Concatenate positions of all trajectories
    x_min, x_max = np.min(all_r[:, 0]), np.max(all_r[:, 0])
    y_min, y_max = np.min(all_r[:, 1]), np.max(all_r[:, 1])
    z_min, z_max = np.min(all_r[:, 2]), np.max(all_r[:, 2])

    x_middle = (x_min + x_max) / 2
    y_middle = (y_min + y_max) / 2
    z_middle = (z_min + z_max) / 2

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)  # Find the maximum range across axes

    # Padding factor for axes
    pad = max_range / 2 * 1.2
    x_range = [x_middle - pad, x_middle + pad]
    y_range = [y_middle - pad, y_middle + pad]
    z_range = [z_middle - pad, z_middle + pad]

    # Fix the scene axes for all frames
    scene_layout = dict(
        xaxis_title='X (m)', 
        yaxis_title='Y (m)', 
        zaxis_title='Z (m)',
        xaxis=dict(
            range=x_range,
        ),
        yaxis=dict(
            range=y_range,
        ),
        zaxis=dict(
            range=z_range,
        ),
        aspectmode='cube',  # Ensure equal scaling for all axes
        camera=dict(eye=dict(x=0.7, y=0.7, z=0.7))
    )

    fig.update_layout(
        title=title,
        scene=scene_layout,
        width=1200, height=800,
        sliders=sliders,
        showlegend=True, hovermode='closest'
    )

    return fig

# def plot_accelerometer(measurements_clean, measurements_noisy, results, sensor_name: SensorName):
#     fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
#     axes_labels = ['X (Body)', 'Y (Body)', 'Z (Body)']
#     colors = ['#e74c3c', '#3498db', '#2ecc71']

#     for ax, dim, label, color in zip(axes, range(3), axes_labels, colors):
#         ax.plot(results.t, measurements_clean[sensor_name.ACCELEROMETER][:, dim], color=color, linewidth=2.5, label='True')
#         ax.scatter(results.t, measurements_noisy[sensor_name.ACCELEROMETER][:, dim], s=8, alpha=0.4, color=color, label='Noisy')
#         ax.set_ylabel(f'{label}\n(m/s²)', fontsize=11, fontweight='bold')
#         ax.grid(True, alpha=0.2, linestyle='--')
#         ax.legend(loc='upper right', fontsize=9)

#     axes[-1].set_xlabel('Time (s)', fontsize=11)
#     fig.suptitle('Accelerometer Measurements', fontsize=13, fontweight='bold', y=0.995)
#     plt.tight_layout()
#     return fig


def plot_accelerometer(measurements_clean, measurements_noisy, results):
    fig, ax = plt.subplots(figsize=(14, 5))
    
    dim_labels = ['X (Body)', 'Y (Body)', 'Z (Body)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for dim, label, color in zip(range(3), dim_labels, colors):
        ax.plot(results.t, measurements_clean["accelerometer"][:, dim], 
                color=color, linewidth=2.5, label=f'{label}', linestyle='-')
        ax.scatter(results.t, measurements_noisy["accelerometer"][:, dim], 
                   s=5, alpha=0.3, color=color)
    
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Acceleration (m/s²)', fontsize=11, fontweight='bold')
    ax.set_title('Accelerometer Measurements', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, ncol=2)
    
    fig.tight_layout()
    return fig


def plot_gyroscope(measurements_clean, measurements_noisy, results):
    fig, ax = plt.subplots(figsize=(14, 5))
    
    dim_labels = ['X (Body Roll)', 'Y (Body Pitch)', 'Z (Body Yaw)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for dim, label, color in zip(range(3), dim_labels, colors):
        ax.plot(results.t, measurements_clean["gyroscope"][:, dim], 
                color=color, linewidth=2.5, label=f'{label}', linestyle='-')
        ax.scatter(results.t, measurements_noisy["gyroscope"][:, dim], 
                   s=5, alpha=0.3, color=color)
    
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Angular Velocity (rad/s)', fontsize=11, fontweight='bold')
    ax.set_title('Gyroscope Measurements', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, ncol=2)
    
    fig.tight_layout()
    return fig



def plot_laser_altimeter(measurements_clean, measurements_noisy, results):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for beam in range(4):
        ax.plot(results.t, measurements_clean["laser_altimeter"][:, beam],
                color=colors[beam], linewidth=2.5, label=f'Beam {beam}', alpha=0.9)
        ax.scatter(results.t, measurements_noisy["laser_altimeter"][:, beam],
                  s=5, alpha=0.15, color=colors[beam])

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Distance to Surface (m)', fontsize=11, fontweight='bold')
    ax.set_title('Laser Altimeter - Four Beam Measurements', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='best', fontsize=10, ncol=4)
    plt.tight_layout()
    return fig

def plot_laser_velocity(measurements_clean, measurements_noisy, results):
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for beam in range(4):
        ax.plot(results.t, measurements_clean["laser_velocity"][:, beam],
                color=colors[beam], linewidth=2.5, label=f'Beam {beam}', alpha=0.9)
        ax.scatter(results.t, measurements_noisy["laser_velocity"][:, beam],
                  s=5, alpha=0.15, color=colors[beam])

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Range Rate (m/s)', fontsize=11, fontweight='bold')
    ax.set_title('Laser Velocity (Doppler) - Four Beam Measurements', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=10, ncol=4)
    plt.tight_layout()
    return fig

def plot_star_tracker(measurements_clean, measurements_noisy, results):
    # fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig, ax = plt.subplots(figsize=(14, 5))

    labels = ['q0 (scalar)', 'q1 (x)', 'q2 (y)', 'q3 (z)']
    colors = ['#9b59b6', '#e74c3c', '#3498db', '#2ecc71']

    for dim, label, color in zip(range(4), labels, colors):
        ax.plot(results.t, measurements_clean["star_tracker"][:, dim],
                color=color, linewidth=2.5, label=label)
        ax.scatter(results.t, measurements_noisy["star_tracker"][:, dim],
                  s=8, alpha=0.3, color=color)
        
    # ax.set_ylabel(label, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.axhline(0, color='k', linestyle='--', alpha=0.2)
    ax.legend(loc='upper right', fontsize=9)

    # axes[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle('Star Tracker - Quaternion Attitude Measurements', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig

def plot_doppler(measurements_clean, measurements_noisy, results):
    fig, ax = plt.subplots(figsize=(14, 5))
    n_sats = measurements_clean["doppler"].shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_sats))

    for sat in range(n_sats):
        ax.plot(results.t, measurements_clean["doppler"][:, sat],
                color=colors[sat], linewidth=2.5, label=f'Sat {sat}', alpha=0.9)
        ax.scatter(results.t, measurements_noisy["doppler"][:, sat],
                  s=5, alpha=0.15, color=colors[sat])

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Doppler Shift (m/s)', fontsize=11, fontweight='bold')
    ax.set_title('Doppler Measurements from Satellites', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    return fig

def plot_range_tracker(measurements_clean, measurements_noisy, results):
    fig, ax = plt.subplots(figsize=(14, 5))
    n_sats = measurements_clean["range_tracker"].shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_sats))

    for sat in range(n_sats):
        ax.plot(results.t, measurements_clean["range_tracker"][:, sat],
                color=colors[sat], linewidth=2.5, label=f'Sat {sat}', alpha=0.9)
        ax.scatter(results.t, measurements_noisy["range_tracker"][:, sat],
                  s=5, alpha=0.15, color=colors[sat])

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Range to Satellite (m)', fontsize=11, fontweight='bold')
    ax.set_title('Range Tracker Measurements to Satellites', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    return fig

def plot_measurements(measurements_clean, measurements_noisy, results, sensor_suite: SensorSuite):
    """Plot only the sensors that are in the suite"""
    plotters = {
        "accelerometer": plot_accelerometer,
        "gyroscope": plot_gyroscope,
        "laser_altimeter": plot_laser_altimeter,
        "laser_velocity": plot_laser_velocity,
        "star_tracker": plot_star_tracker,
        "doppler": plot_doppler,
        "range_tracker": plot_range_tracker,
    }

    print("Sensors in suite:", list(sensor_suite.sensors.keys()))  # debug
    print(type(list(sensor_suite.sensors.keys())[0]))
    print()

    figs = []
    for sensor_name, plotter in plotters.items():
        # print(type(sensor_name))
        # if sensor_name in sensor_suite.sensors:
            fig = plotter(measurements_clean, measurements_noisy, results)
            figs.append(fig)
            plt.show()
            print(f"Plotting {sensor_name}")
    
    return figs


def plot_attitude_relative_vertical(states, t, figsize=(12, 6)):
    """
    Plot the angle between the body's Z-axis and the vertical (downward) direction.
    This shows how tilted the lander is relative to vertical during landing.
    
    Args:
        states: array of shape (n_steps, 13) with [r, v, q, w]
        t: time array
        figsize: figure size
    
    Returns:
        matplotlib figure
    """
    
    # Body Z-axis in body frame
    body_z_axis = np.array([0, 0, 1])
    
    # Vertical direction in inertial frame (downward toward moon center)
    vertical_inertial = np.array([0, 0, 1])
    
    tilt_angles = []
    
    for i in range(len(states)):
        q_B2I = states[i, 6:10]  # quaternion from body to inertial
        
        # Transform body Z-axis to inertial frame
        body_z_inertial = quat_apply(q_B2I, body_z_axis)
        
        # Calculate angle between body Z and vertical
        # Using dot product: cos(theta) = a · b / (|a||b|)
        cos_angle = np.dot(body_z_inertial, vertical_inertial) / (
            np.linalg.norm(body_z_inertial) * np.linalg.norm(vertical_inertial)
        )
        # Clamp to [-1, 1] to avoid numerical issues
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        tilt_angles.append(angle_deg)
    
    tilt_angles = np.array(tilt_angles)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Tilt angle over time
    ax1.plot(t, tilt_angles, 'b-', linewidth=2)
    ax1.fill_between(t, 0, tilt_angles, alpha=0.3)
    ax1.set_ylabel('Tilt Angle (degrees)', fontsize=12)
    ax1.set_title('Lander Attitude: Tilt Angle Relative to Vertical', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3, label='Vertical')
    ax1.legend()
    
    # Altitude vs tilt angle (phase plot)
    altitude = states[:, 2]
    ax2.plot(altitude, tilt_angles, 'r-', linewidth=2)
    ax2.set_xlabel('Altitude (m)', fontsize=12)
    ax2.set_ylabel('Tilt Angle (degrees)', fontsize=12)
    ax2.set_title('Tilt Angle vs Altitude', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    return fig, tilt_angles


def plot_filter_confidence(Sigma_arr, t, figsize=(15, 10)):
    """
    Plot the filter's confidence over time by visualizing the covariance matrix.
    Lower uncertainty = higher confidence.
    
    Args:
        Sigma_arr: array of shape (n_steps, 13, 13) with covariance matrices
        t: time array
        figsize: figure size
    
    Returns:
        matplotlib figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    n_steps = len(Sigma_arr)
    
    # Extract standard deviations for each state component
    pos_std = np.array([np.sqrt(np.diag(Sigma_arr[i, 0:3, 0:3])) for i in range(n_steps)])
    vel_std = np.array([np.sqrt(np.diag(Sigma_arr[i, 3:6, 3:6])) for i in range(n_steps)])
    att_std = np.array([np.sqrt(np.diag(Sigma_arr[i, 6:10, 6:10])) for i in range(n_steps)])
    ang_vel_std = np.array([np.sqrt(np.diag(Sigma_arr[i, 10:13, 10:13])) for i in range(n_steps)])
    
    # Overall metrics
    trace = np.array([np.trace(Sigma_arr[i]) for i in range(n_steps)])
    frobenius = np.array([np.linalg.norm(Sigma_arr[i], 'fro') for i in range(n_steps)])
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Position uncertainty
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(t, pos_std[:, 0], 'r-', label='X', linewidth=1.5)
    ax1.semilogy(t, pos_std[:, 1], 'g-', label='Y', linewidth=1.5)
    ax1.semilogy(t, pos_std[:, 2], 'b-', label='Z', linewidth=1.5)
    ax1.set_ylabel('Std Dev (m)', fontsize=10)
    ax1.set_title('Position Uncertainty', fontsize=11, fontweight='bold')
    ax1.grid(alpha=0.3, which='both')
    ax1.legend(fontsize=9)
    
    # Velocity uncertainty
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(t, vel_std[:, 0], 'r-', label='VX', linewidth=1.5)
    ax2.semilogy(t, vel_std[:, 1], 'g-', label='VY', linewidth=1.5)
    ax2.semilogy(t, vel_std[:, 2], 'b-', label='VZ', linewidth=1.5)
    ax2.set_ylabel('Std Dev (m/s)', fontsize=10)
    ax2.set_title('Velocity Uncertainty', fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3, which='both')
    ax2.legend(fontsize=9)
    
    # Attitude uncertainty
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(t, att_std[:, 0], 'r-', label='q0', linewidth=1.5)
    ax3.semilogy(t, att_std[:, 1], 'g-', label='q1', linewidth=1.5)
    ax3.semilogy(t, att_std[:, 2], 'b-', label='q2', linewidth=1.5)
    ax3.semilogy(t, att_std[:, 3], 'orange', label='q3', linewidth=1.5)
    ax3.set_ylabel('Std Dev', fontsize=10)
    ax3.set_title('Attitude (Quaternion) Uncertainty', fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3, which='both')
    ax3.legend(fontsize=9)
    
    # Angular velocity uncertainty
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(t, ang_vel_std[:, 0], 'r-', label='WX', linewidth=1.5)
    ax4.semilogy(t, ang_vel_std[:, 1], 'g-', label='WY', linewidth=1.5)
    ax4.semilogy(t, ang_vel_std[:, 2], 'b-', label='WZ', linewidth=1.5)
    ax4.set_ylabel('Std Dev (rad/s)', fontsize=10)
    ax4.set_title('Angular Velocity Uncertainty', fontsize=11, fontweight='bold')
    ax4.grid(alpha=0.3, which='both')
    ax4.legend(fontsize=9)
    
    # Overall trace (sum of all variances)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.semilogy(t, trace, 'purple', linewidth=2.5, label='Trace(Σ)')
    ax5.set_ylabel('Trace (overall variance)', fontsize=10)
    ax5.set_title('Overall Filter Confidence (Lower = More Confident)', fontsize=11, fontweight='bold')
    ax5.grid(alpha=0.3, which='both')
    ax5.legend(fontsize=10)
    
    # Frobenius norm (total uncertainty)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.semilogy(t, frobenius, 'darkblue', linewidth=2.5, label='||Σ||_F')
    ax6.set_ylabel('Frobenius Norm', fontsize=10)
    ax6.set_title('Total Uncertainty (Frobenius Norm)', fontsize=11, fontweight='bold')
    ax6.grid(alpha=0.3, which='both')
    ax6.legend(fontsize=10)
    
    # Position uncertainty combined (norm of position variance)
    ax7 = fig.add_subplot(gs[3, 0])
    pos_uncertainty = np.array([np.linalg.norm(pos_std[i]) for i in range(n_steps)])
    vel_uncertainty = np.array([np.linalg.norm(vel_std[i]) for i in range(n_steps)])
    ax7.semilogy(t, pos_uncertainty, 'b-', linewidth=2, label='Position', marker='o', markersize=2, markevery=50)
    ax7.semilogy(t, vel_uncertainty, 'r-', linewidth=2, label='Velocity', marker='s', markersize=2, markevery=50)
    ax7.set_ylabel('Std Dev Magnitude', fontsize=10)
    ax7.set_xlabel('Time (s)', fontsize=10)
    ax7.set_title('Combined Position & Velocity Uncertainty', fontsize=11, fontweight='bold')
    ax7.grid(alpha=0.3, which='both')
    ax7.legend(fontsize=10)
    
    # Confidence indicator: inverse of uncertainty (higher = more confident)
    ax8 = fig.add_subplot(gs[3, 1])
    # Use log scale inverted to show confidence visually
    confidence = 1.0 / (1.0 + trace)  # Normalize to [0, 1]
    ax8.fill_between(t, 0, confidence, alpha=0.5, color='green', label='Confidence')
    ax8.plot(t, confidence, 'g-', linewidth=2)
    ax8.set_ylabel('Confidence Score', fontsize=10)
    ax8.set_xlabel('Time (s)', fontsize=10)
    ax8.set_title('Filter Confidence Level (Higher = More Confident)', fontsize=11, fontweight='bold')
    ax8.set_ylim([0, 1])
    ax8.grid(alpha=0.3)
    ax8.legend(fontsize=10)
    
    fig.suptitle('EKF Covariance Matrix Evolution - Filter Confidence Over Time',
                 fontsize=14, fontweight='bold', y=0.995)

    return fig


def obsv_verbose(x, sensor_suite: SensorSuite, a_m, w_m, Q, sim, env, h=3, show_plot=False):
    """
    Analyze system observability using the observability matrix rank with JAX automatic differentiation.

    Args:
        x: current state estimate [13]
        sensor_suite: SensorSuite with all sensors
        a_m: accelerometer measurement (specific force) [3]
        w_m: gyroscope measurement (angular velocity) [3]
        Q: process noise covariance [13, 13]
        sim: SimParams with simulator configuration
        env: SensorEnvironment with contextual information
        h: number of time steps for observability matrix
        show_plot: whether to plot singular values
    """
    x = jnp.array(x, dtype=float)
    a_m = jnp.array(a_m, dtype=float)
    w_m = jnp.array(w_m, dtype=float)

    O, F_n = [], np.eye(13)

    for _ in range(h):
        H_list = []
        for sensor in sensor_suite.sensors.values():
            jac_fn = jax.jacfwd(lambda s: sensor.measure(s, env))
            H_sensor = np.array(jac_fn(x))
            H_list.append(H_sensor)
        H = np.vstack(H_list)

        O.append(H @ F_n)

        jac_predict = jax.jacfwd(lambda s: ekf_predict_state_only(s, a_m, w_m, sim))
        F = np.array(jac_predict(x))
        x = unitize_state(ekf_predict_state_only(x, a_m, w_m, sim))
        F_n = F_n @ F

    U, S, V = svd(np.vstack(O))
    r = np.sum(S > S[0]*1e-6)

    print(f"Rank {r}/13\n")
    print("OBSERVABLE states:")
    state_names = ["x", "y", "z", "vx", "vy", "vz", "q0", "q1", "q2", "q3", "ωx", "ωy", "ωz"]
    for i in range(r):
        print(f"  Mode {i}: S={S[i]:.2e}")

    print(f"\nUNOBSERVABLE states ({13-r}):")
    for i in range(r, len(S)):
        null_vec = V[i, :]
        contribs = np.abs(null_vec)
        top_idx = np.argsort(contribs)[-1]
        print(f"  Mode {i}: Primary = {state_names[top_idx]} ({null_vec[top_idx]:.3f})")

    if show_plot:
        plt.figure(figsize=(10, 4))
        plt.semilogy(S, 'ko-', linewidth=2, markersize=6)
        plt.axhline(S[0]*1e-6, linewidth=2, label='Rank threshold')
        plt.xlabel('Singular Value Index')
        plt.ylabel('Singular Value')
        plt.title(f'Observability (Rank {r}/13)')
        plt.grid()
        plt.legend()
        plt.show()



def plot_satellites_3d_plotly(sats, lander_position=None, figsize=(1200, 1000)):
    """Interactive 3D Plotly plot of satellite orbits and optional lander trajectory."""
    fig = go.Figure()
    fig.add_trace(moon_surface(radius=R_MOON, offset=np.zeros(3), resolution=30, color='#444444'))

    sat_colors = ['#e74c3c', '#3498db', '#2ecc71']

    for i, sat in enumerate(sats):
        c = sat_colors[i % len(sat_colors)]
        fig.add_trace(go.Scatter3d(
            x=sat.r[:, 0], y=sat.r[:, 1], z=sat.r[:, 2],
            mode='lines', name=f'Satellite {i} Orbit',
            line=dict(color=c, width=4),
        ))
        fig.add_trace(go.Scatter3d(
            x=[sat.r[0, 0]], y=[sat.r[0, 1]], z=[sat.r[0, 2]],
            mode='markers', marker=dict(size=10, color=c, symbol='circle'),
            name=f'Sat {i} Start', showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[sat.r[-1, 0]], y=[sat.r[-1, 1]], z=[sat.r[-1, 2]],
            mode='markers', marker=dict(size=10, color=c, symbol='square'),
            name=f'Sat {i} End', showlegend=False,
        ))

    if lander_position is not None:
        fig.add_trace(go.Scatter3d(
            x=lander_position[:, 0], y=lander_position[:, 1], z=lander_position[:, 2],
            mode='lines', name='Lander Trajectory',
            line=dict(color='#9b59b6', width=5, dash='dash'),
        ))
        fig.add_trace(go.Scatter3d(
            x=[lander_position[0, 0]], y=[lander_position[0, 1]], z=[lander_position[0, 2]],
            mode='markers', marker=dict(size=12, color='#9b59b6', symbol='diamond'),
            name='Lander Start', showlegend=False,
        ))

    all_coords = np.vstack([sat.r for sat in sats] + ([lander_position] if lander_position is not None else []))
    max_range = max(np.max(np.abs(all_coords)) * 1.2, R_MOON * 1.5)

    fig.update_layout(
        title='<b>Satellite Orbits & Lander Trajectory</b>',
        scene=dict(
            xaxis=dict(title='X (m)', range=[-max_range, max_range]),
            yaxis=dict(title='Y (m)', range=[-max_range, max_range]),
            zaxis=dict(title='Z (m)', range=[-max_range, max_range]),
            aspectmode='cube',
        ),
        width=figsize[0], height=figsize[1],
    )
    return fig


def plot_filter_uncertainty_diag(Sigma_arr, t, figsize=(15, 10), use_variance=False) -> Figure:
    """Plot diagonal of EKF covariance matrix (std dev or variance) over time."""
    n_steps = len(Sigma_arr)
    pos_diag     = np.array([np.diag(Sigma_arr[i,  0:3,  0:3]) for i in range(n_steps)])
    vel_diag     = np.array([np.diag(Sigma_arr[i,  3:6,  3:6]) for i in range(n_steps)])
    att_diag     = np.array([np.diag(Sigma_arr[i, 6:10, 6:10]) for i in range(n_steps)])
    ang_vel_diag = np.array([np.diag(Sigma_arr[i, 10:13, 10:13]) for i in range(n_steps)])

    if not use_variance:
        pos_diag, vel_diag, att_diag, ang_vel_diag = (
            np.sqrt(pos_diag), np.sqrt(vel_diag), np.sqrt(att_diag), np.sqrt(ang_vel_diag))

    metric = 'Variance' if use_variance else 'Std Dev'
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('EKF Covariance Diagonal Elements Over Time', fontsize=14, fontweight='bold')

    axs[0, 0].semilogy(t, pos_diag[:, 0], 'r-', label='X', linewidth=2)
    axs[0, 0].semilogy(t, pos_diag[:, 1], 'g-', label='Y', linewidth=2)
    axs[0, 0].semilogy(t, pos_diag[:, 2], 'b-', label='Z', linewidth=2)
    axs[0, 0].set_ylabel(f'{metric} (m{"²" if use_variance else ""})', fontsize=10)
    axs[0, 0].set_title('Position', fontsize=11, fontweight='bold')
    axs[0, 0].grid(alpha=0.3, which='both'); axs[0, 0].legend(fontsize=9)

    axs[0, 1].semilogy(t, vel_diag[:, 0], 'r-', label='VX', linewidth=2)
    axs[0, 1].semilogy(t, vel_diag[:, 1], 'g-', label='VY', linewidth=2)
    axs[0, 1].semilogy(t, vel_diag[:, 2], 'b-', label='VZ', linewidth=2)
    axs[0, 1].set_ylabel(f'{metric} (m/s{"" if not use_variance else "²/s²"})', fontsize=10)
    axs[0, 1].set_title('Velocity', fontsize=11, fontweight='bold')
    axs[0, 1].grid(alpha=0.3, which='both'); axs[0, 1].legend(fontsize=9)

    axs[1, 0].semilogy(t, att_diag[:, 0], 'r-', label='q0', linewidth=2)
    axs[1, 0].semilogy(t, att_diag[:, 1], 'g-', label='q1', linewidth=2)
    axs[1, 0].semilogy(t, att_diag[:, 2], 'b-', label='q2', linewidth=2)
    axs[1, 0].semilogy(t, att_diag[:, 3], color='orange', label='q3', linewidth=2)
    axs[1, 0].set_ylabel(metric, fontsize=10); axs[1, 0].set_xlabel('Time (s)', fontsize=10)
    axs[1, 0].set_title('Attitude (Quaternion)', fontsize=11, fontweight='bold')
    axs[1, 0].grid(alpha=0.3, which='both'); axs[1, 0].legend(fontsize=9)

    axs[1, 1].semilogy(t, ang_vel_diag[:, 0], 'r-', label='ωX', linewidth=2)
    axs[1, 1].semilogy(t, ang_vel_diag[:, 1], 'g-', label='ωY', linewidth=2)
    axs[1, 1].semilogy(t, ang_vel_diag[:, 2], 'b-', label='ωZ', linewidth=2)
    axs[1, 1].set_ylabel(f'{metric} (rad/s{"" if not use_variance else "²/s²"})', fontsize=10)
    axs[1, 1].set_xlabel('Time (s)', fontsize=10)
    axs[1, 1].set_title('Angular Velocity', fontsize=11, fontweight='bold')
    axs[1, 1].grid(alpha=0.3, which='both'); axs[1, 1].legend(fontsize=9)

    plt.tight_layout()
    return fig

def plot_sensor_config_comparison(results_list, t, dropout_times = None, undropout_times = None, alpha = 1):
    """Plot position, velocity, and attitude errors for all sensor configurations."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('EKF Error Comparison Across Sensor Configurations', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for i, result in enumerate(results_list):
        axs[0].semilogy(t, result['pos_error'], label=result['name'], color=colors[i], linewidth=2,alpha=alpha)
    axs[0].set_xlabel('Time (s)', fontsize=11)
    axs[0].set_ylabel('Position Error (m)', fontsize=11)
    axs[0].set_title('Position Error', fontsize=12, fontweight='bold')
    axs[0].grid(True, alpha=0.3, which='both')
    axs[0].legend(fontsize=9, loc='best')
    
    for i, result in enumerate(results_list):
        axs[1].semilogy(t, result['vel_error'], label=result['name'], color=colors[i], linewidth=2,alpha=alpha)
    axs[1].set_xlabel('Time (s)', fontsize=11)
    axs[1].set_ylabel('Velocity Error (m/s)', fontsize=11)
    axs[1].set_title('Velocity Error', fontsize=12, fontweight='bold')
    axs[1].grid(True, alpha=0.3, which='both')
    axs[1].legend(fontsize=9, loc='best')
    
    for i, result in enumerate(results_list):
        axs[2].semilogy(t, result['att_error'], label=result['name'], color=colors[i], linewidth=2,alpha=alpha)
    axs[2].set_xlabel('Time (s)', fontsize=11)
    axs[2].set_ylabel('Attitude Error', fontsize=11)
    axs[2].set_title('Attitude Error', fontsize=12, fontweight='bold')
    axs[2].grid(True, alpha=0.3, which='both')
    axs[2].legend(fontsize=9, loc='best')

    dropout_times = [] if dropout_times is None else dropout_times
    undropout_times = [] if undropout_times is None else undropout_times
    for i in range(3):
        for t_ in dropout_times:
            axs[i].axvline(t_, linestyle="-.", color="tab:red", alpha=0.6)
        for t_ in undropout_times:
            axs[i].axvline(t_, linestyle="-.", color="tab:green", alpha=0.6)
    
    plt.tight_layout()
    
    print("\n" + "="*80)
    print("SENSOR CONFIGURATION COMPARISON - FINAL ERROR METRICS")
    print("="*80)
    for result in results_list:
        pos_final = result['pos_error'][-1]
        vel_final = result['vel_error'][-1]
        att_final = result['att_error'][-1]
        pos_mean = np.mean(result['pos_error'][-100:])
        vel_mean = np.mean(result['vel_error'][-100:])
        print(f"\n{result['name']:25s} | {result['description']}")
        print(f"  Position - Final: {pos_final:8.2f} m  | Mean(last): {pos_mean:8.2f} m")
        print(f"  Velocity - Final: {vel_final:8.4f} m/s | Mean(last): {vel_mean:8.4f} m/s")
        print(f"  Attitude - Final: {att_final:8.4f}")

    return fig
def analyze_ekf_error(results, mu_arr, t_arr, case_name="EKF Performance", save_path=None):
    """
    Streamlined EKF error analysis: position, velocity, attitude, angular rate.
    
    Args:
        results: SimResults object with .states, .t
        mu_arr: [N, 13] array of state estimates from EKF
        t_arr: [N] time array
        case_name: str for figure title
        save_path: optional path to save figure
    
    Returns:
        fig: matplotlib figure
        stats: dict with error metrics
    """
    
    # Component-wise errors
    vel_error_x = mu_arr[:, 3] - results.states[:, 3]
    vel_error_y = mu_arr[:, 4] - results.states[:, 4]
    vel_error_z = mu_arr[:, 5] - results.states[:, 5]
    
    pos_norms = np.linalg.norm(mu_arr[:, 0:3] - results.states[:, 0:3], axis=1)
    vel_norms = np.linalg.norm(mu_arr[:, 3:6] - results.states[:, 3:6], axis=1)
    
    # Angular error: rotation angle between q_true and q_est
    # angle = 2 * arccos(|dot(q_est, q_true)|)
    q_true = results.states[:, 6:10]
    q_est = mu_arr[:, 6:10]
    dot_prod = np.abs(np.sum(q_true * q_est, axis=1))
    dot_prod = np.clip(dot_prod, -1, 1)
    att_error = 2 * np.arccos(dot_prod) * 180 / np.pi  # degrees
    
    # Angular rate error
    w_error = mu_arr[:, 10:13] - results.states[:, 10:13]
    w_norms = np.linalg.norm(w_error, axis=1) * 180 / np.pi  # deg/s
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 8))
    fig.suptitle(f"EKF Error Analysis: {case_name}", fontsize=14, fontweight='bold')
    
    # Position error (log scale)
    pos_error_x = mu_arr[:, 0] - results.states[:, 0]
    pos_error_y = mu_arr[:, 1] - results.states[:, 1]
    pos_error_z = mu_arr[:, 2] - results.states[:, 2]
    
    axes[0, 0].semilogy(t_arr, np.abs(pos_error_x) + 0.1, label='X', linewidth=1.5)
    axes[0, 0].semilogy(t_arr, np.abs(pos_error_y) + 0.1, label='Y', linewidth=1.5)
    axes[0, 0].semilogy(t_arr, np.abs(pos_error_z) + 0.1, label='Z', linewidth=1.5)
    axes[0, 0].set_ylabel('Error (m, log scale)')
    axes[0, 0].set_title('Position Error')
    axes[0, 0].grid(alpha=0.3, which='both')
    axes[0, 0].legend(loc='best')
    
    # Velocity error (log scale)
    axes[0, 1].semilogy(t_arr, np.abs(vel_error_x) + 1e-4, label='Vx', linewidth=1.5)
    axes[0, 1].semilogy(t_arr, np.abs(vel_error_y) + 1e-4, label='Vy', linewidth=1.5)
    axes[0, 1].semilogy(t_arr, np.abs(vel_error_z) + 1e-4, label='Vz', linewidth=1.5)
    axes[0, 1].set_ylabel('Error (m/s, log scale)')
    axes[0, 1].set_title('Velocity Error')
    axes[0, 1].grid(alpha=0.3, which='both')
    axes[0, 1].legend(loc='best')
    
    # Attitude error (degrees)
    axes[1, 0].semilogy(t_arr, np.maximum(att_error, 1e-3), 'red', linewidth=2)
    axes[1, 0].set_ylabel('Error (degrees, log scale)')
    axes[1, 0].set_title('Attitude Error (Rotation Angle)')
    axes[1, 0].grid(alpha=0.3, which='both')
    
    # Angular rate error (deg/s, log scale)
    axes[1, 1].semilogy(t_arr, np.maximum(w_norms, 1e-3), 'blue', linewidth=2)
    axes[1, 1].set_ylabel('Error (°/s, log scale)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Angular Rate Error')
    axes[1, 1].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Compute statistics
    stats = {
        'pos_final': np.linalg.norm(mu_arr[-1, 0:3] - results.states[-1, 0:3]),
        'pos_max': np.max(pos_norms),
        'pos_rms': np.sqrt(np.mean(pos_norms**2)),
        
        'vel_final': np.linalg.norm(mu_arr[-1, 3:6] - results.states[-1, 3:6]),
        'vel_max': np.max(vel_norms),
        'vel_rms': np.sqrt(np.mean(vel_norms**2)),
        'vz_final': np.abs(vel_error_z[-1]),
        
        'att_final': att_error[-1],  # degrees
        'att_max': np.max(att_error),
        
        'rate_final': w_norms[-1],  # deg/s
        'rate_max': np.max(w_norms),
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EKF ERROR ANALYSIS: {case_name}")
    print(f"{'='*60}")
    print(f"\nPosition Error (m):")
    print(f"  Final:   {stats['pos_final']:8.2f}")
    print(f"  Max:     {stats['pos_max']:8.2f}")
    print(f"  RMS:     {stats['pos_rms']:8.2f}")
    
    print(f"\nVelocity Error (m/s):")
    print(f"  Final:   {stats['vel_final']:8.4f}")
    print(f"  Max:     {stats['vel_max']:8.4f}")
    print(f"  RMS:     {stats['vel_rms']:8.4f}")
    print(f"  Vz only: {stats['vz_final']:8.4f}")
    
    print(f"\nAttitude Error (degrees):")
    print(f"  Final:   {stats['att_final']:8.4f}°")
    print(f"  Max:     {stats['att_max']:8.4f}°")
    
    print(f"\nAngular Rate Error (°/s):")
    print(f"  Final:   {stats['rate_final']:8.4f}")
    print(f"  Max:     {stats['rate_max']:8.4f}")
    
    print(f"\nTrajectory: {len(t_arr)} steps, {t_arr[-1]:.1f}s total")
    print(f"{'='*60}\n")
    
    return fig, stats