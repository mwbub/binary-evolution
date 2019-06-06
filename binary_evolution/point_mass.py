import numpy as np
import astropy.units as u
from astropy import constants
from galpy.potential import KeplerPotential

_G = constants.G.to(u.pc**3/u.solMass/u.yr**2).value


class PointMass:
    """
    A class used to determine the perturbation on a Keplerian ring due to a
    distant point mass at the origin.
    """

    def __init__(self, m):
        """Initialize a PointMass.

        Parameters
        ----------
        m : float
            The mass of the PointMass in solar masses.
        """
        self._m = m
        self._pot = KeplerPotential(amp=self._m*u.solMass)

    def potential(self):
        """Return this PointMass's potential.

        Returns
        -------
        pot : galpy.potential.KeplerPotential
            The potential of this PointMass.
        """
        return self._pot

    def m(self):
        """Return this PointMass's mass.

        Returns
        -------
        m : float
            The mass of this PointMass in solar masses.
        """
        return self._m

    def tau(self, kepler_ring, r_mag=None):
        """Return the timescale of the Lidov-Kozai cycles due to this PointMass.

        Parameters
        ----------
        kepler_ring : KeplerRing
            The Keplerian ring undergoing the Lidov-Kozai oscillations.
        r_mag : float, optional
            The magnitude of the position vector of kepler_ring in pc. If not
            provided, the initial position of kepler_ring will be used instead.

        Returns
        -------
        tau : float
            The timescale in years.

        Notes
        -----
        Because tau depends on the position vector r, it is not a constant.
        Thus, tau should be taken only as an approximation of the timescale,
        which varies with the orbit of kepler_ring about the Galactic centre.
        """
        if r_mag is None:
            r_mag = np.sum(kepler_ring.r()[:2]**2)**0.5

        return ((2 * kepler_ring.m()**0.5 * r_mag**3) /
                (3 * _G**0.5 * self._m * kepler_ring.a(pc=True)**1.5))

    def lk_func(self, kepler_ring):
        """Return a function used to evaluate Lidov-Kozai derivatives.

        Parameters
        ----------
        kepler_ring : KeplerRing
            The Keplerian ring undergoing the Lidov-Kozai oscillations.

        Returns
        -------
        func : callable
            A function used to evaluate the derivatives of e and j due to
            Lidov-Kozai oscillations. The calling signature is func(t, e, j, r),
            where t is the time step, e and j are the eccentricity and
            dimensionless angular momentum vectors, and r is the position vector
            of the barycentre in Cartesian coordinates. The return value is a
            tuple (de, dj), where de and dj are arrays of shape (3,)
            representing the derivatives of e and j.
        """
        return lambda t, e, j, r: self._lk_derivatives(kepler_ring, e, j, r)

    def _lk_derivatives(self, kepler_ring, e, j, r):
        """Compute the derivatives of e and j from the Lidov-Kozai cycles of
        this PointMass.

        Parameters
        ----------
        kepler_ring : KeplerRing
            The Keplerian ring undergoing the Lidov-Kozai oscillations.
        e : ndarray
            The eccentricity vector, of the form [ex, ey, ez].
        j : ndarray
            The dimensionless angular momentum vector, of the form [jx, jy, jz].
        r : ndarray
            Position vector of the barycentre in Cartesian coordinates, of the
            form [x, y, z] in pc.

        Returns
        -------
        de : ndarray
            An array of shape (3,) representing the derivative of e.
        dj : ndarray
            An array of shape (3,) representing the derivative of j.
        """
        r_mag = np.sum(r**2)**0.5
        r_hat = r/r_mag
        tau = self.tau(kepler_ring, r_mag)

        # Cross products
        j_cross_e = np.cross(j, e)
        j_cross_r = np.cross(j, r_hat)
        e_cross_r = np.cross(e, r_hat)

        # Dot products
        r_dot_e = np.dot(r_hat, e)
        r_dot_j = np.dot(r_hat, j)

        # Dimensionless derivatives
        de = 2 * j_cross_e - 5 * r_dot_e * j_cross_r + r_dot_j * e_cross_r
        dj = r_dot_j * j_cross_r - 5 * r_dot_e * e_cross_r

        # Multiply by -tau^-1
        de /= -tau
        dj /= -tau

        return de, dj
