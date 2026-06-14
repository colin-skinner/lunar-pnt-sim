from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import json
from numpyencoder import NumpyEncoder

class TrajectoryData(BaseModel):
    class Config:
        extra = 'allow'
        arbitrary_types_allowed=True

    seed: int
    
    # Simulation
    dt: float
    T: float
    nsteps: int
    t: np.ndarray

    # Initial conditions
    state0: np.ndarray
    mass_kg: float
    I: np.ndarray

    # Optimization (n=6, m=6)
    s_bar: np.ndarray = Field(default_factory=lambda: np.array([])) # [nsteps, n] r,v
    u_bar: np.ndarray = Field(default_factory=lambda: np.array([])) # [nsteps, m] F,tau
    K_fb: np.ndarray = Field(default_factory=lambda: np.array([])) # [nsteps, m, n]
    k_ff: np.ndarray = Field(default_factory=lambda: np.array([])) # [nsteps, m]

    # Into body frame
    force_B: np.ndarray
    torque_B: np.ndarray



t = np.zeros((300))
pos = np.zeros((300,3))
vel = np.zeros((300,3))

data = TrajectoryData(time=t)
with open('traj.json', 'w') as f:
    json.dump(data.model_dump(mode='python'), f, cls=NumpyEncoder)