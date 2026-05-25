import numpy as np
import plotly.graph_objects as go
from .sim.math.quaternion import quat_apply
from .constants import R_MOON

# Define specific colors
x_axis_color = 'red'
y_axis_color = 'green'
z_axis_color = 'blue'

def moon_surface(xx, yy, zz):
    """Returns go.Surface of Moon"""
    return go.Surface(x=xx, y=yy, z=zz, colorscale=[[0, '#333333'], [1, '#555555']],
                             showscale=False, name='Moon', hoverinfo='skip')

def add_lander(position):
    """Add lander marker to the figure."""
    return go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]],
                                 mode='markers', marker=dict(size=6, color='black'),
                                 name='Lander', hoverinfo='skip')

def add_gradient_trajectory(r, t, colorscale='Viridis'):
    """Return go.Scatter3d of trajectory colored by time."""
    return go.Scatter3d(
        x=r[:, 0], y=r[:, 1], z=r[:, 2],
        mode='lines',
        line=dict(color=t, colorscale=colorscale, width=3, showscale=False),
        name='Trajectory'  # Name for trajectory
    )

def add_solid_trajectory(r, color='red'):
    """Return go.Scatter3d of trajectory colored by time."""
    return go.Scatter3d(
        x=r[:, 0], y=r[:, 1], z=r[:, 2],
        mode='lines',
        line=dict(color=color, width=2, showscale=False),
        name='Solid Trajectory'  # Name for solid trajectory
    )

def get_body_axes(r, q, axis_size):
    """Returns go.Scatter3d traces for body axes based on current position and orientation."""
    traces = []
    for i, (name, color, vec) in enumerate([('X-axis', x_axis_color, [1, 0, 0]),
                                             ('Y-axis', y_axis_color, [0, 1, 0]),
                                             ('Z-axis', z_axis_color, [0, 0, 1])]):
        axis = quat_apply(q, vec) * axis_size
        traces.append(go.Scatter3d(
            x=[r[0], r[0] + axis[0]],
            y=[r[1], r[1] + axis[1]],
            z=[r[2], r[2] + axis[2]],
            mode='lines', line=dict(color=color, width=3),
            name=name, hoverinfo='skip'
        ))
    return traces

def visualize_trajectory(
    states: np.ndarray,
    t: np.ndarray = None,
    dt: float = 0.1,
    axis_scale: float = 1000.0,
    title: str = "Lunar Descent Trajectory"
):
    """
    Simple interactive 3D trajectory visualizer.
    
    Args:
        states: [N, 13] array (r, v, q, omega)
        t: time array (auto-generated if None)
        dt: time step in seconds
        axis_scale: length of body axis vectors (meters)
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
    xx, yy = np.meshgrid(np.linspace(-10000, 10000, 5), np.linspace(-10000, 10000, 5))
    zz = np.full_like(xx, r[0, 2])  # Use the initial altitude
    moon_surface_trace = moon_surface(xx, yy, zz)
    fig.add_trace(moon_surface_trace)

    # Lander marker
    fig.add_trace(add_lander(r[0]))

    # Trajectory colored by time
    traj = add_gradient_trajectory(r, t)
    fig.add_trace(traj)
    
    # Solid trajectory for visibility
    solid_traj = add_solid_trajectory(r, color='gold')  # Solid gold trajectory
    fig.add_trace(solid_traj)

    # Body axes setup 
    traj_scale = np.max(np.linalg.norm(r - r[0], axis=1))  # max distance from start
    axis_size = min(axis_scale, traj_scale * 0.1)  # 10% of trajectory extent or user-specified
    fig.add_traces(get_body_axes(r[0], q[0], axis_size))

    # Animation frames
    frames = []
    for step in range(n_steps):
        pos = r[step]
        frame_data = [
            moon_surface_trace,
            add_lander(r[step]),
            traj,  # Include the gradient trajectory
            solid_traj,  # Add solid trajectory
            *get_body_axes(r[step], q[step], axis_size)  # Include body axes for the current state
        ]
        frames.append(go.Frame(data=frame_data, name=str(step)))  # Append the frame data
    fig.frames = frames

    # Slider and Updatemenus
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
            gridcolor='rgba(255, 255, 255, 0.1)',  # Color of the grid lines
            zerolinecolor='rgba(255, 255, 255, 0.3)',  # Color of the zero line on x-axis
            titlefont=dict(color='white'),  # Title font color for x-axis
            tickfont=dict(color='white'),  # Tick font color for x-axis
            linecolor='white'  # Color of the x-axis line
        ),
        yaxis=dict(
            range=y_range,
            gridcolor='rgba(255, 255, 255, 0.1)',  # Color of the grid lines
            zerolinecolor='rgba(255, 255, 255, 0.3)',  # Color of the zero line on y-axis
            titlefont=dict(color='white'),  # Title font color for y-axis
            tickfont=dict(color='white'),  # Tick font color for y-axis
            linecolor='white'  # Color of the y-axis line
        ),
        zaxis=dict(
            range=z_range,
            gridcolor='rgba(255, 255, 255, 0.1)',  # Color of the grid lines
            zerolinecolor='rgba(255, 255, 255, 0.3)',  # Color of the zero line on z-axis
            titlefont=dict(color='white'),  # Title font color for z-axis
            tickfont=dict(color='white'),  # Tick font color for z-axis
            linecolor='white'  # Color of the z-axis line
        ),
        aspectmode='cube',  # Ensure equal scaling for all axes
        # bgcolor='rgb(0, 0, 0, 1)',  # Set scene background to black
        camera=dict(eye=dict(x=0.7, y=0.7, z=0.7))
    )

    fig.update_layout(
        title=f'{title}<br>t = {t[0]:.2f}s',
        scene=scene_layout,
        width=1200, height=800,
        sliders=sliders,
        showlegend=True, hovermode='closest',
    )

    return fig