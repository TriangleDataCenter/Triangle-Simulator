from scipy.integrate import quad
from scipy.optimize import root_scalar
import numpy as np 

from Triangle.Constants import H0, OmegaM, MPC, C 

def LuminosityDistance(z, h0 = H0, om = OmegaM):
    """ 
    Returns: luminosity distance in [m]
    """
    invE = lambda x: 1. / np.sqrt(om * (1. + x) ** 3 + 1. - om)
    return C * 1e-3 * (1. + z) / h0 * MPC * quad(invE, 0, z)[0]

def z_dl(dl, h0 = H0, om = OmegaM):
    temp = lambda z: LuminosityDistance(z, h0, om) - dl
    sol = root_scalar(temp, method="bisect", bracket=(0, 15), xtol = 1e-7)
    return sol.root

def Hubble(z, h0 = H0, om = OmegaM):
    return h0 * np.sqrt(om * (1. + z) ** 3 + 1. - om)