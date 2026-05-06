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