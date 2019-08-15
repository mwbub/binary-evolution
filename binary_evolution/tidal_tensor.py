import numpy as np
from galpy.potential import ttensor, Potential
from galpy.util.bovy_conversion import freq_in_Gyr, time_in_Gyr

# Factors for conversion from galpy internal units
_pc = 8000
_yr = time_in_Gyr(220, 8) * 1e+9
_freq2_in_yr = (freq_in_Gyr(220, 8) / 1e+9)**2


class TidalTensor:
    """
    A helper class used to enable the calculation of tidal tensors in galpy for
    ellipsoidal triaxial potentials.
    """

    def __init__(self, pot):
        """Initialize a TidalTensor instance.

        Parameters
        ----------
        pot : galpy.potential.Potential or list of Potentials
            The potentials for which the tidal tensor will be evaluated.
        """
        if isinstance(pot, Potential):
            pot = [pot]
        self._pot = pot

        # List of tidal tensor functions customized for each potential
        tts = []
        for pot in self._pot:
            if hasattr(pot, '_2ndderiv_xyz'):
                tts.append(lambda p, x, y, z, t: _ttensor_manual(p, x, y, z))
            else:
                tts.append(_ttensor_galpy)
        self._tts = tts

    def __call__(self, x, y, z, t=0):
        """Evaluate the tidal tensor at a point.

        Parameters
        ----------
        x, y, z : float
            Cartesian coordinates in pc.
        t : float, optional
            The time of evaluation in years.

        Returns
        -------
        ttensor : ndarray
            The tidal tensor in yr^-2
        """
        return sum([tt(p, x, y, z, t) for p, tt in zip(self._pot, self._tts)])


# noinspection PyProtectedMember, PyUnresolvedReferences
def _ttensor_manual(pot, x, y, z):
    """Calculate the tidal tensor manually at a point.

    Parameters
    ----------
    pot : galpy.potential.Potential
        The potential to evaluate. This must possess a pot._2ndderiv_xyz method.
    x, y, z : float
        Cartesian coordinates in pc.

    Returns
    -------
    ttensor : ndarray
        The tidal tensor in yr^-2
    """
    # Translate to galpy internal units
    x /= _pc
    y /= _pc
    z /= _pc

    txx = pot._2ndderiv_xyz(x, y, z, 0, 0)
    tyy = pot._2ndderiv_xyz(x, y, z, 1, 1)
    tzz = pot._2ndderiv_xyz(x, y, z, 2, 2)
    txy = pot._2ndderiv_xyz(x, y, z, 0, 1)
    txz = pot._2ndderiv_xyz(x, y, z, 0, 2)
    tyz = pot._2ndderiv_xyz(x, y, z, 1, 2)
    tyx = txy
    tzx = txz
    tzy = tyz

    tt = np.array([[txx, txy, txz], [tyx, tyy, tyz], [tzx, tzy, tzz]])
    tt *= pot._amp * _freq2_in_yr

    return tt


def _ttensor_galpy(pot, x, y, z, t):
    """Calculate the tidal tensor via galpy's built in tidal tensor method.

    Parameters
    ----------
    pot : galpy.potential.Potential or list of Potentials
        The potential to evaluate.
    x, y, z : float
        Cartesian coordinates in pc.
    t : float
        The time of evaluation in years.

    Returns
    -------
    ttensor : ndarray
        The tidal tensor in yr^-2
    """
    R = (x**2 + y**2)**0.5
    phi = np.arctan2(y, x)

    tt = -ttensor(pot, R/_pc, z/_pc, phi=phi, t=t/_yr, use_physical=False)
    tt *= _freq2_in_yr

    return tt
