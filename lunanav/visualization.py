import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .sim.math.quaternion import quat_apply
from .constants import R_MOON



def visualize_trajectory(
    states: np.ndarray,
    force: np.ndarray = None,
    torques: np.ndarray = None,
    t: np.ndarray = None,
    dt: float = 0.1,
    moon_radius: float = R_MOON,  # km
    initial_altitude: float = 10e3,
    show_sensors: bool = True,
    sensor_range: float = 5000.0,  # meters
    lander_size: float = 500.0,  # meters (scaled for visibility)
    axis_scale: float = 2000.0,  # meters
    title: str = "Lunar Descent Trajectory"
):
    """
    Interactive 3D visualizer for powered descent.
    
    Args:
        states: [N, 13] array (r, v, q, omega)
        force: [N, 3] array optional (fx, fy, fz)
        torques: [N, 3] array optional (tx, ty, tz)
        t: time array, auto-generated if None
        dt: time step in seconds
        moon_radius: moon radius in meters
        initial_altitude: height of moon surface plane in world coords
        show_sensors: whether to draw sensor LOS vectors
        sensor_range: length of sensor visualization rays
        lander_size: scale of lander marker
        axis_scale: length of body axis vectors
        title: plot title
    
    Returns:
        plotly Figure
    """
    
    n_steps = len(states)
    
    # Auto-generate time if not provided
    if t is None:
        t = np.arange(n_steps) * dt
    
    # Extract states
    r = states[:, 0:3]   # position
    v = states[:, 3:6]   # velocity
    q = states[:, 6:10]  # quaternion
    w = states[:, 10:13] # angular velocity
    
    # Initialize figure
    fig = go.Figure()
    
    # === TRAJECTORY ===
    fig.add_trace(go.Scatter3d(
        x=r[:, 0],
        y=r[:, 1],
        z=r[:, 2],
        mode='lines',
        name='Trajectory',
        line=dict(color='white', width=2),
        hovertemplate='<b>Trajectory</b><br>x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<extra></extra>'
    ))
    
    # === MOON SURFACE (plane at z = initial_altitude - moon_radius) ===
    moon_z = initial_altitude - moon_radius
    
    # Create a square plane
    plane_size = 50000  # meters
    xx, yy = np.meshgrid(
        np.linspace(-plane_size, plane_size, 10),
        np.linspace(-plane_size, plane_size, 10)
    )
    zz = np.full_like(xx, moon_z)
    
    fig.add_trace(go.Surface(
        x=xx,
        y=yy,
        z=zz,
        colorscale=[[0, '#333333'], [1, '#555555']],
        showscale=False,
        name='Moon Surface',
        hoverinfo='skip'
    ))
    
    # === FORCES (if provided) ===
    if force is not None:
        force_scale = 0.001  # scale for visibility
        
        # Sample force at regular intervals for clarity
        force_interval = max(1, n_steps // 20)
        force_indices = np.arange(0, n_steps, force_interval)
        
        for idx in force_indices:
            pos = r[idx]
            
            # Normalize and scale
            f_mag = np.linalg.norm(force)
            if f_mag > 1:
                f_norm = force / f_mag * force_scale * f_mag
                
                fig.add_trace(go.Scatter3d(
                    x=[pos[0], pos[0] + f_norm[0]],
                    y=[pos[1], pos[1] + f_norm[1]],
                    z=[pos[2], pos[2] + f_norm[2]],
                    mode='lines',
                    line=dict(color='orange', width=3),
                    name='Force' if idx == force_indices[0] else '',
                    showlegend=int(idx == force_indices[0]) == 1,
                    hoverinfo='skip'
                ))
    
    # === SENSOR LOS VECTORS ===
    if show_sensors:
        sensor_interval = max(1, n_steps // 15)
        sensor_indices = np.arange(0, n_steps, sensor_interval)
        
        # Downward-facing altimeter (body -z)
        for idx in sensor_indices:
            pos = r[idx]
            quat = q[idx]
            sensor_dir_body = np.array([0, 0, -1])  # body frame
            sensor_dir_inertial = quat_apply(quat, sensor_dir_body)
            sensor_end = pos + sensor_dir_inertial * sensor_range
            
            fig.add_trace(go.Scatter3d(
                x=[pos[0], sensor_end[0]],
                y=[pos[1], sensor_end[1]],
                z=[pos[2], sensor_end[2]],
                mode='lines',
                line=dict(color='cyan', width=1, dash='dash'),
                name='Altimeter' if idx == sensor_indices[0] else '',
                showlegend=int(idx == force_indices[0]) == 1,
                hoverinfo='skip'
            ))
    
    # === LANDER MARKER (sphere at current position) ===
    # This will be the marker that moves with the slider
    fig.add_trace(go.Scatter3d(
        x=[r[0, 0]],
        y=[r[0, 1]],
        z=[r[0, 2]],
        mode='markers',
        marker=dict(size=8, color='yellow', symbol='diamond'),
        name='Lander',
        hovertemplate='<b>Lander</b><br>x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<extra></extra>'
    ))
    
    # === BODY AXES (initial) ===
    # These will be updated by frames
    for axis_idx, (axis_name, axis_color, axis_vec) in enumerate([
        ('X', 'red', [1, 0, 0]),
        ('Y', 'green', [0, 1, 0]),
        ('Z', 'blue', [0, 0, 1])
    ]):
        pos = r[0]
        quat = q[0]
        axis_inertial = quat_apply(quat, axis_vec) * axis_scale
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0], pos[0] + axis_inertial[0]],
            y=[pos[1], pos[1] + axis_inertial[1]],
            z=[pos[2], pos[2] + axis_inertial[2]],
            mode='lines',
            line=dict(color=axis_color, width=4),
            name=f'Axis {axis_name}',
            hoverinfo='skip'
        ))
    
    # === CREATE FRAMES FOR ANIMATION ===
    frames = []
    
    for step in range(n_steps):
        pos = r[step]
        quat = q[step]
        
        # Update lander position
        lander_data = [
            go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]])
        ]
        
        # Update body axes
        for axis_vec in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
            axis_inertial = quat_apply(quat, axis_vec) * axis_scale
            lander_data.append(go.Scatter3d(
                x=[pos[0], pos[0] + axis_inertial[0]],
                y=[pos[1], pos[1] + axis_inertial[1]],
                z=[pos[2], pos[2] + axis_inertial[2]]
            ))
        
        frames.append(go.Frame(
            data=lander_data,
            name=str(step),
            layout=go.Layout(
                title_text=f'{title}<br>t = {t[step]:.2f} s'
            )
        ))
    
    fig.frames = frames
    
    # === SLIDER ===
    sliders = [{
        'active': 0,
        'yanchor': 'top',
        'y': 0,
        'xanchor': 'left',
        'x': 0.1,
        'len': 0.9,
        'transition': {'duration': 0},
        'pad': {'b': 10, 't': 50},
        'currentvalue': {
            'prefix': 'Time: ',
            'visible': True,
            'xanchor': 'right',
            'suffix': ' s'
        },
        'steps': [
            {
                'args': [
                    [str(step)],
                    {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }
                ],
                'method': 'animate',
                'label': f'{t[step]:.2f}'
            }
            for step in range(n_steps)
        ]
    }]
    
    # === PLAY/PAUSE BUTTONS ===
    updatemenus = [{
        'buttons': [
            {
                'args': [None, {
                    'frame': {'duration': 50, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': 'Pause',
                'method': 'animate'
            }
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

    # Add padding (10% of range)
    x_min, x_max = np.min(r[:, 0]), np.max(r[:, 0])
    y_min, y_max = np.min(r[:, 1]), np.max(r[:, 1])
    z_min, z_max = np.min(r[:, 2]), np.max(r[:, 2])
    
    x_pad = (x_max - x_min) * 0.1 if x_max > x_min else 1000
    y_pad = (y_max - y_min) * 0.1 if y_max > y_min else 1000
    z_pad = (z_max - z_min) * 0.1 if z_max > z_min else 1000
    
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]
    z_range = [z_min - z_pad, z_max + z_pad]
    
    # === LAYOUT ===
    fig.update_layout(
        title=f'{title}<br>t = {t[0]:.2f} s',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode='data',
            camera=dict(
                eye=dict(x=0.7, y=0.7, z=0.7),
                center=dict(x=0, y=0, z=0)
            )
        ),
        width=1200,
        height=800,
        sliders=sliders,
        updatemenus=updatemenus,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.7)')
    )
    
    return fig