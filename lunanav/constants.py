from __future__ import annotations
import numpy as np

########################################
#               Angle
########################################

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 1 / DEG_TO_RAD

DEG_TO_ARCSEC = 3600
ARCSEC_TO_DEG = 1/3600

ARCSEC_TO_RAD = ARCSEC_TO_DEG * DEG_TO_RAD 
RAD_TO_ARCSEC = 1 / ARCSEC_TO_RAD 

########################################
#               Moon Info
########################################

GM_MOON = 4902.800118e9 
"""m^3/s^2"""

R_MOON = 1737.4e3
"""m"""

MOON_3_VEC = lambda x: np.tile([0,0,R_MOON],(x,1))
MOON_13_VEC = lambda x: np.tile([0,0,R_MOON,0,0,0,0,0,0,0,0,0,0],(x,1))

########################################
#               NOIse
########################################

sigma_accel = .01      # m/s^2
sigma_gyro = 1e-3      # rad/s
sigma_los = 50        # m (super noisy!)
sigma_los_vel = 1     # m/s (super noisy!)
sigma_star = 1e-3      # unitless
sigma_doppler = 0.1      # m/s
sigma_sat_range_tracker = 10        # m (super noisy!)
