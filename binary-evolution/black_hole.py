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
