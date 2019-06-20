import numpy as np
import astropy.units as u
from galpy.orbit import Orbit
from galpy.potential import vcirc, vesc


def ecc_to_vel(pot, ecc, r, tol=1e-4):
    """Calculate the tangential velocity required to achieve an eccentricity.

    Parameters
    ----------
    pot : galpy.potential.Potential or list of Potentials
        The potential containing the orbiting object.
    ecc : float
        The desired eccentricity.
    r : array_like
        Position vector of the orbiting object in Galactocentric cylindrical
        coordinates, of the form [R, z, phi] in [pc, pc, rad].
    tol : float
        Error tolerance between the desired and the achieved eccentricity.

    Returns
    -------
    v_phi : float
        Tangential velocity in km/s required to achieve ecc, assuming that the
        object begins at pericentre with 0 vertical velocity.
    """
    if not 0 <= ecc < 1:
        raise ValueError("Eccentricity must be between 0 and 1")

    R, z, phi = r
    v_high = vesc(pot, R*u.pc, vo=220, ro=8)
    v_low = vcirc(pot, R*u.pc, phi=phi*u.rad, vo=220, ro=8)
    v_cur = v_low
    ecc_cur = 0

    while np.abs(ecc - ecc_cur) > tol:
        if ecc_cur > ecc:
            v_high = v_cur
            v_cur = (v_cur + v_low) / 2
        else:
            v_low = v_cur
            v_cur = (v_cur + v_high) / 2

        orb = Orbit(vxvv=[R*u.pc, 0*u.km/u.s, v_cur*u.km/u.s, z*u.pc,
                          0*u.km/u.s, phi*u.rad])

        try:
            ecc_cur = orb.e(pot=pot, analytic=True)
        except ValueError:
            orb_R = orb.R(use_physical=False)
            P = orb_R * 2 * np.pi / vcirc(pot, orb_R, use_physical=False)
            t = np.linspace(0, 10*P, 1000)
            orb.integrate(t, pot)
            ecc_cur = orb.e()

    return v_cur
