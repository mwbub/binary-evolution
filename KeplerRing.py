import numpy as np
import astropy.units as u
from astropy import constants
from galpy.orbit import Orbit
from galpy.potential import ttensor
from scipy.integrate import solve_ivp

_G = constants.G.to(u.pc**3/u.M_sun/u.yr**2).value


class KeplerRing:
    """
    A class used to evolve a Keplerian ring using vectorial formalism.
    """

    def __init__(self, e, j, r, v, m, a):
        """Initialize a Keplerian ring.

        Parameters
        ----------
        e : array_like
            Initial e vector, of the form [ex, ey, ez].
        j : array_like
            Initial j vector, of the form [jx, jy, jz].
        r : array_like
            Initial position of the barycentre in Galactocentric cylindrical
            coordinates, of the form [R, z, phi] in [pc, pc, rad].
        v : array_like
            Initial velocity of the barycentre in Galactocentric cylindrical
            coordinates, of the form [v_R, v_z, v_phi] in km/s.
        a : float
            Semi-major axis of the ring in AU.
        m : float
            Total mass of the ring in solar masses.
        """
        self._e = e
        self._j = j
        self._r = r
        self._v = v
        self._a = (a*u.au).to(u.pc).value
        self._m = m
        self._t = None
        self._ej = None

        R, z, phi = self._r
        v_R, v_z, v_phi = self._v
        self._orb = Orbit(vxvv=[R*u.pc, v_R*u.km/u.s, v_phi*u.km/u.s, z*u.pc,
                                v_z*u.km/u.s, phi*u.rad])

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

    def r(self):
        """Return the time evolution of the barycentre position vector.

        Returns
        -------
        Array of r vectors, with the same length as the time array used for
        integration. If this KeplerRing has never been integrated, returns the
        initial j vector instead. Units are [pc, pc, rad].
        """
        raise NotImplementedError  # TODO: Implement this

    def v(self):
        """Return the time evolution of the barycentre velocity vector.

        Returns
        -------
        Array of v vectors, with the same length as the time array used for
        integration. If this KeplerRing has never been integrated, returns the
        initial j vector instead. Units are in km/s.
        """
        raise NotImplementedError  # TODO: Implement this

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
        The mass in solar masses.
        """
        return self._m

    def a(self):
        """Return the semi-major axis of this KeplerRing.

        Returns
        -------
        The semi-major axis in AU.
        """
        return (self._a*u.pc).to(u.au).value


class KeplerRingError(Exception):
    pass
