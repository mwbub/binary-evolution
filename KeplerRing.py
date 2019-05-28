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
        self._e = e
        self._j = j
        self._r = r
        self._m = m
        self._a = a
        self._t = None
        self._ej = None
        self._orb = Orbit(vxvv=self._r)

    def e(self):
        """Return the time evolution of the e vector.

        Returns
        -------
        Array of e vectors, with the same length as the time array used for
        integration. If this KeplerRing has never been integrated, returns the
        initial e vector instead.
        """
        if self._ej is None:
            return self._e
        return self._ej[:, :3]

    def j(self):
        """Return the time evolution of the j vector.

        Returns
        -------
        Array of j vectors, with the same length as the time array used for
        integration. If this KeplerRing has never been integrated, returns the
        initial j vector instead.
        """
        if self._ej is None:
            return self._j
        return self._ej[:, 3:]

    def t(self):
        """Return the time array used to integrate this KeplerRing.

        Returns
        -------
        Array of time steps.
        """
        if self._t is None:
            raise KeplerRingError("Use KeplerRing.integrate() before using "
                                  "KeplerRing.t()")
        return self._t

    def m(self):
        """Return the total mass of this KeplerRing.

        Returns
        -------
        The mass.
        """
        return self._m

    def a(self):
        """Return the semi-major axis of this KeplerRing.

        Returns
        -------
        The semi-major axis.
        """
        return self._a


class KeplerRingError(Exception):
    pass
