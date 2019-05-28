import numpy as np
from scipy.integrate import solve_ivp
from galpy.potential import ttensor
from galpy.orbit import Orbit


class KeplerRing:
    """
    A class used to evolve a Keplerian ring using vectorial formalism.
    """

    def __init__(self, e, j, r, m, a):
        """Initialize a Keplerian ring.

        Parameters
        ----------
        e : array_like
            Initial e vector, of the form [ex, ey, ez].
        j : array_like
            Initial j vector, of the form [jx, jy, jz].
        r : array_like
            Initial position and velocity of the barycentre in Galactocentric
            cylindrical coordinates, of the form [R, vR, vT, z, vz, phi].
        m : float
            Total mass of the ring.
        a : float
            Semi-major axis of the ring.
        """
        self.e = e
        self.j = j
        self.r = r
        self.m = m
        self.a = a
        self._t = None
        self._ej = None
        self._orb = Orbit(vxvv=self.r)
