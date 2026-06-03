"""Simple save/load for trajectories and EKF results."""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class Trajectory:
    """iLQR trajectory solution"""
    s_bar: np.ndarray  # Nominal states [N+1, 13]
    u_bar: np.ndarray  # Nominal controls [N, m]
    dt: float  # Time step (s)
    T: float  # Total duration (s)
    nsteps: int  # Number of steps
    state0: np.ndarray  # Initial state [13]
    mass_kg: float  # Vehicle mass (kg)
    I: np.ndarray  # Inertia matrix [3, 3]


@dataclass
class SensorData:
    """Truth and noisy measurements for one sensor over the simulation."""
    truth: np.ndarray   # Noiseless measurements [N, dim]
    noisy: np.ndarray   # Noisy measurements [N, dim], NaN where invalid


@dataclass
class SimResult:
    """Full simulation run: true trajectory + per-sensor measurements."""
    s_true: np.ndarray                   # True simulated states [N+1, 13]
    t_arr: np.ndarray                    # Time array [N+1]
    measurements: dict                   # {sensor_name: SensorData}

    # Sim parameters (mirrors Trajectory for standalone use)
    dt: float
    nsteps: int
    mass_kg: float
    I: np.ndarray


@dataclass
class EKFResult:
    """EKF filter results"""
    mu_arr: np.ndarray  # State estimates [N, 13]
    Sigma_arr: np.ndarray  # Covariance [N, 13, 13]
    t_arr: np.ndarray  # Time array [N]

    # Simulation parameters (for reproducibility)
    dt: float
    nsteps: int
    mass_kg: float
    I: np.ndarray


def save_trajectory(traj: Trajectory, filepath: str) -> None:
    """Save trajectory to JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "s_bar": traj.s_bar,
        "u_bar": traj.u_bar,
        "dt": float(traj.dt),
        "T": float(traj.T),
        "nsteps": int(traj.nsteps),
        "state0": traj.state0,
        "mass_kg": float(traj.mass_kg),
        "I": traj.I,
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)


def load_trajectory(filepath: str) -> Trajectory:
    """Load trajectory from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    return Trajectory(
        s_bar=np.array(data["s_bar"]),
        u_bar=np.array(data["u_bar"]),
        dt=float(data["dt"]),
        T=float(data["T"]),
        nsteps=int(data["nsteps"]),
        state0=np.array(data["state0"]),
        mass_kg=float(data["mass_kg"]),
        I=np.array(data["I"]),
    )


def save_sim_result(result: SimResult, filepath: str) -> None:
    """Save full simulation result to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    meas_data = {}
    for name, sd in result.measurements.items():
        meas_data[name] = {"truth": sd.truth, "noisy": sd.noisy}
    data = {
        "s_true": result.s_true,
        "t_arr": result.t_arr,
        "measurements": meas_data,
        "dt": float(result.dt),
        "nsteps": int(result.nsteps),
        "mass_kg": float(result.mass_kg),
        "I": result.I,
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)


def load_sim_result(filepath: str) -> SimResult:
    """Load full simulation result from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    measurements = {
        name: SensorData(truth=np.array(sd["truth"]), noisy=np.array(sd["noisy"]))
        for name, sd in data["measurements"].items()
    }
    return SimResult(
        s_true=np.array(data["s_true"]),
        t_arr=np.array(data["t_arr"]),
        measurements=measurements,
        dt=float(data["dt"]),
        nsteps=int(data["nsteps"]),
        mass_kg=float(data["mass_kg"]),
        I=np.array(data["I"]),
    )


def save_ekf_result(result: EKFResult, filepath: str) -> None:
    """Save EKF result to JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mu_arr": result.mu_arr,
        "Sigma_arr": result.Sigma_arr,
        "t_arr": result.t_arr,
        "dt": float(result.dt),
        "nsteps": int(result.nsteps),
        "mass_kg": float(result.mass_kg),
        "I": result.I,
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)


def load_ekf_result(filepath: str) -> EKFResult:
    """Load EKF result from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    return EKFResult(
        mu_arr=np.array(data["mu_arr"]),
        Sigma_arr=np.array(data["Sigma_arr"]),
        t_arr=np.array(data["t_arr"]),
        dt=float(data["dt"]),
        nsteps=int(data["nsteps"]),
        mass_kg=float(data["mass_kg"]),
        I=np.array(data["I"]),
    )
