import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from .sim.quaternion import quat_apply
from .constants import R_MOON

# Define specific colors
x_axis_color = 'red'
y_axis_color = 'green'
z_axis_color = 'blue'

# def moon_surface(xx, yy, zz):
#     """Returns go.Surface of Moon"""
#     return go.Surface(x=xx, y=yy, z=zz, colorscale=[[0, '#333333'], [1, '#555555']],
#                              showscale=False, name='Moon', hoverinfo='skip')

def moon_surface(radius=R_MOON, offset: np.ndarray = np.zeros(3), resolution=25, color='#444444'):
    """Returns go.Surface of Moon as a sphere"""
    # Create sphere using spherical coordinates
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v)) + offset[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + offset[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + offset[2]
    
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
    *, 
    title: str = "Lunar Descent Trajectories",
    other_vecs: dict = None,
    show_body_axes: bool = True,
    show_lander: bool = True,
    downsample_rate: int = 5  # For downsampling the number of plotted frames
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

    if isinstance(trajectories, np.ndarray):
        trajectories = [trajectories]  # Wrap in list if single trajectory provided
    
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
        moon_surface_trace = moon_surface(radius=R_MOON, offset = [0,0,-R_MOON], resolution=100)
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


def plot_measurements(measurements_clean: dict, measurements_noisy: dict, results, sensor_suite, lander, figsize: tuple = (15, 14)):
    """
    Plot all sensor measurements comparing clean vs noisy data.

    Args:
        measurements_clean: Dict[SensorName] -> array of shape (n_steps, meas_dim)
        measurements_noisy: Dict[SensorName] -> array of shape (n_steps, meas_dim)
        results: SimResults object with states, forces, torques, and time
        sensor_suite: SensorSuite object with all sensors
        lander: RigidBody object (for mass)
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure with 5 subplots (one per sensor type)
    """
    from .sim.sensors import SensorName

    fig, axes = plt.subplots(5, 1, figsize=figsize)

    # Accelerometer (3 channels)
    accel_clean = measurements_clean[SensorName.ACCELEROMETER]
    accel_noisy = measurements_noisy[SensorName.ACCELEROMETER]
    accel_true = results.force_N / lander.mass_kg
    for j in range(3):
        axes[0].plot(results.t, accel_true[:, j], 'k-', alpha=0.5, linewidth=1.5)
        axes[0].plot(results.t, accel_noisy[:, j], '.', markersize=1, alpha=0.4)
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title("Accelerometer: True (black) vs Noisy (colored dots)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(["X (truth)", "Y (truth)", "Z (truth)"], loc="upper right")

    # Gyroscope (3 channels)
    gyro_clean = measurements_clean[SensorName.GYROSCOPE]
    gyro_noisy = measurements_noisy[SensorName.GYROSCOPE]
    gyro_true = results.states[:, 10:13]
    for j in range(3):
        axes[1].plot(results.t, gyro_true[:, j], 'k-', alpha=0.5, linewidth=1.5)
        axes[1].plot(results.t, gyro_noisy[:, j], '.', markersize=1, alpha=0.4)
    axes[1].set_ylabel("Angular Velocity (rad/s)")
    axes[1].set_title("Gyroscope: True (black) vs Noisy (colored dots)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(["X (truth)", "Y (truth)", "Z (truth)"], loc="upper right")

    # Laser altimeter (4 channels)
    laser_alt_clean = measurements_clean[SensorName.LASER_ALTIMETER]
    laser_alt_noisy = measurements_noisy[SensorName.LASER_ALTIMETER]
    for j in range(4):
        axes[2].plot(results.t, laser_alt_clean[:, j], '-', alpha=0.7, linewidth=1.5, label=f"LOS {j+1} (truth)")
        axes[2].plot(results.t, laser_alt_noisy[:, j], '.', markersize=1, alpha=0.3)
    axes[2].set_ylabel("Distance (m)")
    axes[2].set_title("Laser Altimeter: Clean (lines) vs Noisy (dots)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right", ncol=4, fontsize=8)

    # Laser velocity (4 channels)
    laser_vel_clean = measurements_clean[SensorName.LASER_VELOCITY]
    laser_vel_noisy = measurements_noisy[SensorName.LASER_VELOCITY]
    for j in range(4):
        axes[3].plot(results.t, laser_vel_clean[:, j], '-', alpha=0.7, linewidth=1.5, label=f"LOS {j+1} (truth)")
        axes[3].plot(results.t, laser_vel_noisy[:, j], '.', markersize=1, alpha=0.3)
    axes[3].set_ylabel("Range Rate (m/s)")
    axes[3].set_title("Laser Velocity: Clean (lines) vs Noisy (dots)")
    axes[3].grid(alpha=0.3)
    axes[3].legend(loc="upper right", ncol=4, fontsize=8)

    # Star tracker (quaternion components)
    star_clean = measurements_clean[SensorName.STAR_TRACKER]
    star_noisy = measurements_noisy[SensorName.STAR_TRACKER]
    star_true = results.states[:, 6:10]
    for j in range(4):
        axes[4].plot(results.t, star_true[:, j], 'k-', alpha=0.5, linewidth=1.5)
        axes[4].plot(results.t, star_noisy[:, j], '.', markersize=1, alpha=0.4)
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("Quaternion Component")
    axes[4].set_title("Star Tracker (Attitude): True (black) vs Noisy (colored dots)")
    axes[4].grid(alpha=0.3)
    axes[4].legend(["q0 (truth)", "q1 (truth)", "q2 (truth)", "q3 (truth)"], loc="upper right")

    plt.tight_layout()

    return fig


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
    from lunanav.sim.quaternion import quat_apply
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Body Z-axis in body frame
    body_z_axis = np.array([0, 0, 1])
    
    # Vertical direction in inertial frame (downward toward moon center)
    vertical_inertial = np.array([0, 0, -1])
    
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