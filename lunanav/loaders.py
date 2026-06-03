"""Simple save/load for trajectories, sim results, and EKF results.

All functions accept and return plain dicts. On load, JSON lists are
converted back to numpy arrays for known array fields.

Trajectory dict keys:
    s_bar     [N+1, 13]   nominal states
    u_bar     [N, m]      nominal controls
    dt        float
    T         float
    nsteps    int
    state0    [13]
    mass_kg   float
    I         [3, 3]

SimResult dict keys:
    s_true        [N+1, 13]
    t_arr         [N+1]
    dt            float
    nsteps        int
    mass_kg       float
    I             [3, 3]
    measurements  dict of {sensor_name: sensor_dict}
                  sensor_name: str, e.g. "laser_altimeter", "doppler"
                  sensor_dict keys:
                      truth  [N, dim]   noiseless measurements at each timestep
                      noisy  [N, dim]   noisy measurements, NaN where invalid/out of range
                  dims by sensor:
                      accelerometer   [N, 3]
                      gyroscope       [N, 3]
                      laser_altimeter [N, 4]   (4 LOS beams)
                      laser_velocity  [N, 4]   (4 LOS beams)
                      star_tracker    [N, 4]   (quaternion)
                      doppler         [N, n_sats]
                      range_tracker   [N, n_sats]

EKFResult dict keys:
    mu_arr    [N, 13]
    Sigma_arr [N, 13, 13]
    t_arr     [N]
    dt        float
    nsteps    int
    mass_kg   float
    I         [3, 3]
"""

import json
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy arrays to lists."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save(data: dict, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)


def _load(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)


_TRAJECTORY_ARRAYS = {"s_bar", "u_bar", "state0", "I"}
_SIM_ARRAYS = {"s_true", "t_arr", "I"}
_EKF_ARRAYS = {"mu_arr", "Sigma_arr", "t_arr", "I"}


def save_trajectory(traj: dict, filepath: str) -> None:
    _save(traj, filepath)


def load_trajectory(filepath: str) -> dict:
    data = _load(filepath)
    for key in _TRAJECTORY_ARRAYS:
        if key in data:
            data[key] = np.array(data[key])
    return data


def save_sim_result(result: dict, filepath: str) -> None:
    _save(result, filepath)


def load_sim_result(filepath: str) -> dict:
    data = _load(filepath)
    for key in _SIM_ARRAYS:
        if key in data:
            data[key] = np.array(data[key])
    for sensor_data in data.get("measurements", {}).values():
        sensor_data["truth"] = np.array(sensor_data["truth"])
        sensor_data["noisy"] = np.array(sensor_data["noisy"])
    return data


def save_ekf_result(result: dict, filepath: str) -> None:
    _save(result, filepath)


def load_ekf_result(filepath: str) -> dict:
    data = _load(filepath)
    for key in _EKF_ARRAYS:
        if key in data:
            data[key] = np.array(data[key])
    return data
