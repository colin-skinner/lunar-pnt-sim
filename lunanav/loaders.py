"""Simple save/load for trajectories, sim results, and EKF results.

Trajectory:
    s_bar       [N+1, 13]   nominal states from iLQR
    u_bar       [N, m]      nominal controls from iLQR
    dt          float       timestep (s)
    T           float       total duration (s)
    nsteps      int         number of steps
    state0      [13]        initial state
    mass_kg     float
    I           [3, 3]      inertia matrix
    t           [N+1]       time array
    force       [N, 3]      body-frame force (N)
    torque      [N, 3]      body-frame torque (N·m)

SimResult:
    trajectory_file  str         filename of the Trajectory used to generate this
    s_arr      [N+1, 13]   true simulated states
    t_arr       [N+1]       time array
    dt          float
    nsteps      int
    mass_kg     float
    I           [3, 3]
    measurements  dict of {sensor_name: {"truth": [N, dim], "noisy": [N, dim]}}
                  sensor_name is a string (SensorName.value)
                  noisy contains NaN where sensor is invalid/out of range
                  dims by sensor:
                      accelerometer   [N, 3]
                      gyroscope       [N, 3]
                      laser_altimeter [N, 4]   (4 LOS beams)
                      laser_velocity  [N, 4]   (4 LOS beams)
                      star_tracker    [N, 4]   (quaternion)
                      doppler         [N, n_sats]
                      range_tracker   [N, n_sats]

EKFResult:
    trajectory_file  str         filename of the Trajectory used to generate this
    mu_arr      [N, 13]     state estimates
    Sigma_arr   [N, 13, 13] covariance matrices
    t_arr       [N]         time array
    dt          float
    nsteps      int
    mass_kg     float
    I           [3, 3]
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Trajectory:
    s_bar: np.ndarray
    u_bar: np.ndarray
    dt: float
    T: float
    nsteps: int
    state0: np.ndarray
    mass_kg: float
    I: np.ndarray
    t: np.ndarray
    force: np.ndarray
    torque: np.ndarray


@dataclass
class SimResult:
    trajectory_file: str
    s_arr: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    t_arr: np.ndarray
    dt: float
    nsteps: int
    mass_kg: float
    I: np.ndarray
    measurements: dict  # {sensor_name: {"truth": ndarray, "noisy": ndarray}}


@dataclass
class EKFResult:
    trajectory_file: str
    mu_arr: np.ndarray
    Sigma_arr: np.ndarray
    t_arr: np.ndarray
    dt: float
    nsteps: int
    mass_kg: float
    I: np.ndarray


def save_trajectory(traj: Trajectory, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "s_bar": traj.s_bar.tolist(), 
        "u_bar": traj.u_bar.tolist(),
        "dt": traj.dt, 
        "T": traj.T, 
        "nsteps": traj.nsteps,
        "state0": traj.state0.tolist(), 
        "mass_kg": traj.mass_kg, 
        "I": traj.I.tolist(),
        "t": traj.t.tolist(), 
        "force": traj.force.tolist(), 
        "torque": traj.torque.tolist(),
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_trajectory(filepath: str) -> Trajectory:
    with open(filepath) as f:
        d = json.load(f)
    return Trajectory(
        s_bar=np.array(d["s_bar"]),
        u_bar=np.array(d["u_bar"]),
        dt=float(d["dt"]),
        T=float(d["T"]),
        nsteps=int(d["nsteps"]),
        state0=np.array(d["state0"]),
        mass_kg=float(d["mass_kg"]),
        I=np.array(d["I"]),
        t=np.array(d["t"]),
        force=np.array(d["force"]),
        torque=np.array(d["torque"]),
    )


def save_sim_result(result: SimResult, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "trajectory_file": result.trajectory_file,
        "s_arr": result.s_arr.tolist(),
        "force": result.force.tolist(),
        "torque": result.torque.tolist(),
        "t_arr": result.t_arr.tolist(),
        "dt": result.dt,
        "nsteps": result.nsteps,
        "mass_kg": result.mass_kg,
        "I": result.I.tolist(),
        "measurements": {
            name: {"truth": sd["truth"].tolist(), "noisy": sd["noisy"].tolist()}
            for name, sd in result.measurements.items()
        },
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_sim_result(filepath: str) -> SimResult:
    with open(filepath) as f:
        d = json.load(f)
    measurements = {
        name: {"truth": np.array(sd["truth"]), "noisy": np.array(sd["noisy"])}
        for name, sd in d["measurements"].items()
    }
    return SimResult(
        trajectory_file=d["trajectory_file"],
        s_arr=np.array(d["s_arr"]),
        force=np.array(d["force"]),
        torque=np.array(d["torque"]),
        t_arr=np.array(d["t_arr"]),
        dt=float(d["dt"]),
        nsteps=int(d["nsteps"]),
        mass_kg=float(d["mass_kg"]),
        I=np.array(d["I"]),
        measurements=measurements,
    )


def save_ekf_result(result: EKFResult, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "trajectory_file": result.trajectory_file,
        "mu_arr": result.mu_arr.tolist(),
        "Sigma_arr": result.Sigma_arr.tolist(),
        "t_arr": result.t_arr.tolist(),
        "dt": result.dt,
        "nsteps": result.nsteps,
        "mass_kg": result.mass_kg,
        "I": result.I.tolist(),
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_ekf_result(filepath: str) -> EKFResult:
    with open(filepath) as f:
        d = json.load(f)
    return EKFResult(
        trajectory_file=d["trajectory_file"],
        mu_arr=np.array(d["mu_arr"]),
        Sigma_arr=np.array(d["Sigma_arr"]),
        t_arr=np.array(d["t_arr"]),
        dt=float(d["dt"]),
        nsteps=int(d["nsteps"]),
        mass_kg=float(d["mass_kg"]),
        I=np.array(d["I"]),
    )
