import numpy as np
import plotly.graph_objects as go
from .sim.math.quaternion import quat_apply
from .constants import R_MOON

# Define specific colors
x_axis_color = 'red'
y_axis_color = 'green'
z_axis_color = 'blue'

def visualize_trajectory(
    states: np.ndarray,
    t: np.ndarray = None,
    dt: float = 0.1,
    force: np.ndarray = None,
    axis_scale: float = 1000.0,
    moon_radius: float = R_MOON,
    initial_altitude: float = 10e3,
    title: str = "Lunar Descent Trajectory"
):
    """
    Simple interactive 3D trajectory visualizer.
    Args:
        states: [N, 13] array (r, v, q, omega)
        t: time array (auto-generated if None)
        dt: time step in seconds
        force: [N, 3] array optional (only fx, fy, fz)
        axis_scale: length of body axis vectors (meters)
        moon_radius: moon radius (meters)
        initial_altitude: starting altitude (meters)
        title: plot title
    Returns:
        plotly Figure
    """
    
    n_steps = len(states)
    if t is None:
        t = np.arange(n_steps) * dt
    
    # Extract states
    r = states[:, 0:3]
    q = states[:, 6:10]
    
    fig = go.Figure()
    
    # Moon surface
    moon_z = initial_altitude - moon_radius
    xx, yy = np.meshgrid(np.linspace(-10000, 10000, 5), np.linspace(-10000, 10000, 5))
    zz = np.full_like(xx, moon_z)
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale=[[0, '#333333'], [1, '#555555']], 
                             showscale=False, name='Moon', hoverinfo='skip'))
    
    # Trajectory colored by time (only for visualization purpose)
    fig.add_trace(go.Scatter3d(
        x=r[:, 0], y=r[:, 1], z=r[:, 2],
        mode='lines',
        line=dict(color=t, colorscale='Viridis', width=3, showscale=False)  # Keep this to visualize in play
    ))
    
    # # Force vectors (if provided)
    # if force is not None:
    #     force_interval = max(1, n_steps // 15)
    #     for idx in range(0, n_steps, force_interval):
    #         f = force[idx]
    #         f_mag = np.linalg.norm(f)
    #         if f_mag > 1:
    #             f_norm = f / f_mag * 500  # fixed scale for visibility
    #             pos = r[idx]
    #             fig.add_trace(go.Scatter3d(
    #                 x=[pos[0], pos[0] + f_norm[0]],
    #                 y=[pos[1], pos[1] + f_norm[1]],
    #                 z=[pos[2], pos[2] + f_norm[2]],
    #                 mode='lines', line=dict(color='orange', width=2),
    #                 name='Force' if idx == 0 else '', showlegend=(idx == 0), hoverinfo='skip'
    #             ))
    
    # Lander marker
    fig.add_trace(go.Scatter3d(x=[r[0, 0]], y=[r[0, 1]], z=[r[0, 2]],
                               mode='markers', marker=dict(size=6, color='black'),
                               name='Lander', hoverinfo='skip'))
    
    # Body axes (scaled by overall trajectory size, not current state)
    traj_scale = np.max(np.linalg.norm(r - r[0], axis=1))  # max distance from start
    axis_size = min(axis_scale, traj_scale * 0.1)  # 10% of trajectory extent, or user-specified, whichever is smaller
    
    for i, (name, color, vec) in enumerate([('X', x_axis_color, [1, 0, 0]), ('Y', y_axis_color, [0, 1, 0]), ('Z', z_axis_color, [0, 0, 1])]):
        axis = quat_apply(q[0], vec) * axis_size
        fig.add_trace(go.Scatter3d(
            x=[r[0, 0], r[0, 0] + axis[0]],
            y=[r[0, 1], r[0, 1] + axis[1]],
            z=[r[0, 2], r[0, 2] + axis[2]],
            mode='lines', line=dict(color=color, width=3),
            name=f'{name}-axis', hoverinfo='skip'
        ))
    
    # Animation frames
    frames = []
    for step in range(n_steps):
        pos = r[step]
        frame_data = [go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]],
                                    mode='markers', marker=dict(size=6, color='black'))]

        # Add the colored trajectory to each frame
        frame_data.append(go.Scatter3d(x=r[:, 0], y=r[:, 1], z=r[:, 2],
                                        mode='lines', line=dict(color=t, colorscale='Viridis', width=3, showscale=False)))

        # Only add current body axes for the current state
        for vec, color, name in zip([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
                                    [x_axis_color, y_axis_color, z_axis_color], 
                                    ['X-axis', 'Y-axis', 'Z-axis']):
            axis = quat_apply(q[step], vec) * axis_scale
            frame_data.append(go.Scatter3d(
                x=[pos[0], pos[0] + axis[0]],
                y=[pos[1], pos[1] + axis[1]],
                z=[pos[2], pos[2] + axis[2]],
                mode='lines', line=dict(color=color, width=3),
                name=f'{name}', hoverinfo='skip'
            ))

        frames.append(go.Frame(data=frame_data, name=str(step)))  # no layout at all
    fig.frames = frames
    
    # Slider and Updatemenus (remains unchanged)
    sliders = [{'active': 0, 'yanchor': 'top', 'y': 0, 'xanchor': 'left', 'x': 0.1, 'len': 0.9,
                'currentvalue': {'prefix': 'Time: ', 'suffix': ' s', 'visible': True},
                'steps': [{'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                           'method': 'animate', 'label': f'{t[i]:.1f}'} for i in range(n_steps)]
    }]
    
    # Auto-scale to the maximum trajectory extent
    x_min, x_max = np.min(r[:, 0]), np.max(r[:, 0])
    y_min, y_max = np.min(r[:, 1]), np.max(r[:, 1])
    z_min, z_max = np.min(r[:, 2]), np.max(r[:, 2])

    x_middle = (x_min + x_max) / 2
    y_middle = (y_min + y_max) / 2
    z_middle = (z_min + z_max) / 2

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)  # Find the maximum range across axes

    # Padding factor
    pad = max_range/2 * 1.2
    x_range = [x_middle - pad, x_middle + pad]
    y_range = [y_middle - pad, y_middle + pad]
    z_range = [z_middle - pad, z_middle + pad]

    # Fix the scene axes for all frames (update this section)
    scene_layout = dict(
        xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
        xaxis=dict(range=x_range), 
        yaxis=dict(range=y_range), 
        zaxis=dict(range=z_range),
        aspectmode='cube',  # Ensure equal scaling for all axes
        camera=dict(eye=dict(x=0.7, y=0.7, z=0.7))
    )
    
    fig.update_layout(
        title=f'{title}<br>t = {t[0]:.2f}s',
        scene=scene_layout,
        width=1200, height=800,
        sliders=sliders,
        showlegend=True, hovermode='closest'
    )
    
    return fig