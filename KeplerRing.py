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
        # Initial conditions
        self._e0 = e
        self._j0 = j
        self._r0 = r
        self._v0 = v
        self._a = (a*u.au).to(u.pc).value
        self._m = m

        # Result arrays
        self._t = None   # Time array
        self._ej = None  # Combined e/j array
        self._r = None   # Position vector array
        self._v = None   # Velocity vector array

    def integrate(self, t, pot=None, func=None, alt_pot=None):
        """Integrate the orbit of this KeplerRing.

        Parameters
        ----------
        t : array_like
            Array of times at which to output, in years. Must be 1D and sorted.
        pot : galpy.potential.Potential, optional
            A potential used to integrate the orbit. This potential's tidal
            tensor will be used to evolve the e and j vectors. If not provided,
            you must provide both a func and alt_pot parameter to integrate the
            e/j vectors and barycentre, respectively.
        func : callable, optional
            An additional term to add to the derivatives of the e and j vectors.
            The calling signature is func(t, e, j, r) where t is the time step,
            x = (e, j) is the combined e/j vector, and r is the position vector
            of the barycentre. The return value must be a tuple (de, dj), where
            de and dj are arrays of shape (1, 3) representing the derivatives of
            e and j.
        alt_pot : galpy.potential.Potential, optional
            An additional potential used to integrate the barycentre position,
            but not to evolve the e and j vectors.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def e(self):
        """Return the time evolution of the e vector.

        Returns
        -------
        Array of e vectors, with the same length as the time array used for
        integration. If this KeplerRing has never been integrated, returns the
        initial e vector instead.
        """
        if self._ej is None:
            return self._e0
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
            return self._j0
        return self._ej[:, 3:]

    def r(self):
        """Return the time evolution of the barycentre position vector.

        Returns
        -------
        Array of r vectors, with the same length as the time array used for
        integration. If this KeplerRing has never been integrated, returns the
        initial j vector instead. Units are [pc, pc, rad].
        """
        if self._r is None:
            return self._r0
        return self._r

    def v(self):
        """Return the time evolution of the barycentre velocity vector.

        Returns
        -------
        Array of v vectors, with the same length as the time array used for
        integration. If this KeplerRing has never been integrated, returns the
        initial j vector instead. Units are km/s.
        """
        if self._v is None:
            return self._v0
        return self._v

    def t(self):
        """Return the time array used to integrate this KeplerRing.

        Returns
        -------
        Array of time steps in years.
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
