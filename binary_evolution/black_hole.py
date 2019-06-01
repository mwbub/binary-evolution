import numpy as np
import astropy.units as u
from astropy import constants
from galpy.potential import KeplerPotential

_G = constants.G.to(u.pc**3/u.solMass/u.yr**2).value


class BlackHole:
    """
    A class used to determine the perturbation on a Keplerian ring due to a
    distant point mass (i.e., a supermassive black hole).
    """

    def __init__(self, m):
        """Initialize a black hole.

        Parameters
        ----------
        m : float
            The mass of the black hole in solar masses.
        """
        self._m = m
        self._pot = KeplerPotential(amp=self._m*u.solMass)

    def potential(self):
        """Return this BlackHole's potential.

        Returns
        -------
        pot : galpy.potential.KeplerPotential
            The potential of this BlackHole.
        """
        return self._pot

    def m(self):
        """Return this BlackHole's mass.

        Returns
        -------
        m : float
            The mass of this BlackHole in solar masses.
        """
        return self._m

    def tau(self, kepler_ring, r_mag=None):
        """Return the timescale of the Lidov-Kozai cycles due to this BlackHole.

        Parameters
        ----------
        kepler_ring : KeplerRing
            The Keplerian ring undergoing the Lidov-Kozai oscillations.
        r_mag : float, optional
            The magnitude of the position vector of kepler_ring in pc. If not
            provided, the initial position of kepler_ring will be used instead.

        Returns
        -------
        tau : The timescale in years.

        Notes
        -----
        Because tau depends on the position vector r, it is not a constant.
        Thus, tau should be taken only as an approximation of the timescale,
        which varies with the orbit of kepler_ring about the Galactic centre.
        """
        if r_mag is None:
            r_mag = np.sum(kepler_ring.r()**2)**0.5

        return ((2 * kepler_ring.m()**0.5 * r_mag**3) /
                (3 * _G**0.5 * self._m * kepler_ring.a(pc=True)**1.5))
