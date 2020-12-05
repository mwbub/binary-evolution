import numpy as np
import astropy.units as u
from scipy import optimize
from galpy.orbit import Orbit
from galpy.actionAngle import UnboundError
from galpy.potential import evaluaterforces, evaluatePotentials, PotentialError
from galpy.util.bovy_conversion import time_in_Gyr

# Factors for conversion from galpy internal units
_pc = 8000
_kms = 220
_yr = time_in_Gyr(220, 8) * 1e+9

# Exceptions thrown when using galpy.actionAngle
_action_angle_error = (ValueError, ZeroDivisionError, NotImplementedError,
                       TypeError, PotentialError)


def v_circ(pot, r):
    """Calculate the approximate circular velocity at r.

    Parameters
    ----------
    pot : galpy.potential.Potential or list of Potentials
        The potential used to calculate the circular velocity
    r : array_like
        Initial position vector of the desired orbit in Galactocentric
        cylindrical coordinates, of the form [R, z, phi] in [pc, pc, rad].

    Returns
    -------
    vc : float
        The approximate circular velocity at r.
    """
    R, z, phi = r
    r_mag = (R**2 + z**2)**0.5
    r_force = -evaluaterforces(pot, R/_pc, z/_pc, phi=phi, use_physical=False)
    vc = (r_mag / _pc * r_force) ** 0.5
    return vc * _kms


def v_esc(pot, r):
    """Calculate the escape velocity at r

    Parameters
    ----------
    pot : galpy.potential.Potential or list of Potentials
        The potential used to calculate the escape velocity.
    r : array_like
        Initial position vector of the desired orbit in Galactocentric
        cylindrical coordinates, of the form [R, z, phi] in [pc, pc, rad].

    Returns
    -------
    ve : float
        The escape velocity at r.
    """
    R, z, phi = r
    E_min = evaluatePotentials(pot, R/_pc, z/_pc, phi=phi, use_physical=False)
    E_max = evaluatePotentials(pot, 1e+12, 0, phi=phi, use_physical=False)
    ve = (2 * (E_max - E_min))**0.5
    return ve * _kms


def period(pot, r, v, method='dop853_c'):
    """Calculate the azimuthal period of an orbit via orbit integration.

    Parameters
    ----------
    pot : galpy.potential.Potential or list of Potentials
        The potential containing the orbit.
    r : array_like
        Initial position in Galactocentric cylindrical coordinates, of the form
        [R, z, phi] in [pc, pc, rad].
    v : array_like
        Initial velocity in Galactocentric cylindrical coordinates, of the form
        [v_R, v_z, v_phi] in km/s.
    method : str, optional
        Method used to integrate the orbit. See the documentation for
        galpy.orbit.Orbit.integrate for available options.

    Returns
    -------
    T : float
        The azimuthal period in years.

    Notes
    -----
    This method is unreliable for computing the periods of orbits with
    velocities close to the escape velocity or to 0. It works best when the
    orbital velocity is comparable to the circular velocity.
    """
    # Set up the orbit
    R, z, phi = r
    v_R, v_z, v_phi = v
    orb = Orbit(vxvv=[R/_pc, v_R/_kms, v_phi/_kms, z/_pc, v_z/_kms, phi])

    # Compute the circular period
    vc = v_circ(pot, r) / _kms
    r_mag = (R**2 + z**2)**0.5 / _pc
    Tc = 2 * np.pi * r_mag / vc

    # Number of periods to integrate and time resolution
    num_periods = 10
    resolution = 100

    tries = 0
    max_tries = 3
    while tries < max_tries:
        tries += 1

        # Integrate the orbit for num_periods circular periods
        t = np.linspace(0, Tc * num_periods, num_periods * resolution)
        orb.integrate(t, pot, method=method)

        # Find phase wrapping events
        phi_orb = orb.phi(t) % (2 * np.pi)
        phase_wrap_events = np.abs(phi_orb[1:] - phi_orb[:-1]) > np.pi
        num_events = np.sum(phase_wrap_events)

        # Try again if too few phase wrapping events occurred
        if (num_events < 2) or (num_events < 5 and tries < max_tries):
            num_periods *= 10
            continue

        # Compute T by calculating the spacing between phase wrapping events
        t_phase_wrap = t[1:][phase_wrap_events]
        T = np.mean(t_phase_wrap[1:] - t_phase_wrap[:-1])

        # Try again if there are fewer than 20 time steps per period
        if T / (t[1] - t[0]) < 20:
            resolution *= 10
            continue

        return T * _yr

    raise RuntimeError("Failed to calculate the period")


def ecc_to_vel(pot, ecc, r, tol=1e-4):
    """Calculate the tangential velocity required to achieve an eccentricity.

    Parameters
    ----------
    pot : galpy.potential.Potential or list of Potentials
        The potential containing the orbiting object.
    ecc : float
        The desired eccentricity.
    r : array_like
        Pericentre position vector of the orbiting object in Galactocentric
        cylindrical coordinates, of the form [R, z, phi] in [pc, pc, rad].
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

    # Assume maximum velocity is the escape velocity at R
    v_high = v_esc(pot, r)

    if ecc != 0:
        # Calculate the circular velocity via a recursive call
        v_low = ecc_to_vel(pot, 0, r, tol=tol)

        # Return the approximate v_circular if the user requests a very low ecc
        ecc_low = get_ecc(pot, r, [0, 0, v_low])
        if ecc <= ecc_low:
            return v_low

        # Calculate the desired velocity, between v_circular and v_escape
        return optimize.brentq(lambda v: get_ecc(pot, r, [0, 0, v]) - ecc,
                               v_low, v_high, xtol=tol, maxiter=1000)

    # Calculate the approximate circular velocity by minimizing eccentricity
    v_low = v_circ(pot, r) / 2
    return optimize.minimize_scalar(lambda v: get_ecc(pot, r, [0, 0, v]),
                                    method='bounded', bounds=[v_low, v_high],
                                    options={'xatol': tol, 'maxiter': 1000}).x


def get_ecc(pot, r, v):
    """Calculate the eccentricity of an orbit.

    Parameters
    ----------
    pot : galpy.potential.Potential or list of Potentials
        The potential of the orbit.
    r : array_like
        Initial position in Galactocentric cylindrical coordinates, of the form
        [R, z, phi] in [pc, pc, rad].
    v : array_like
        Initial velocity in Galactocentric cylindrical coordinates, of the form
        [v_R, v_z, v_phi] in km/s.

    Returns
    -------
    ecc : float
        The eccentricity.
    """
    # Set up the orbit
    R, z, phi = r
    v_R, v_z, v_phi = v
    orb = Orbit(vxvv=[R*u.pc, v_R*u.km/u.s, v_phi*u.km/u.s, z*u.pc,
                      v_z*u.km/u.s, phi*u.rad])

    try:
        # Calculate the eccentricity analytically (via action-angles)
        ecc = orb.e(pot=pot, analytic=True)
    except _action_angle_error:
        # Integrate for 50 circular periods
        vc = v_circ(pot, r) / _kms
        r_mag = (R**2 + z**2)**0.5 / _pc
        Tc = 2 * np.pi * r_mag / vc
        t = np.linspace(0, 50*Tc, 1000)
        orb.integrate(t, pot, method='dop853_c')

        # Calculate the eccentricity numerically
        ecc = orb.e()
    except UnboundError:
        return 1

    if np.isnan(ecc):
        return 1

    return ecc


def get_a(pot, r, v):
    """Calculate the semi-major axis of an orbit.

    Parameters
    ----------
    pot : galpy.potential.Potential or list of Potentials
        The potential of the orbit.
    r : array_like
        Initial position in Galactocentric cylindrical coordinates, of the form
        [R, z, phi] in [pc, pc, rad].
    v : array_like
        Initial velocity in Galactocentric cylindrical coordinates, of the form
        [v_R, v_z, v_phi] in km/s.


    Returns
    -------
    a : float
        The semi-major axis.
    """
    # Set up the orbit and calculate the period
    R, z, phi = r
    v_R, v_z, v_phi = v
    P = period(pot, r, v) / _yr
    orb = Orbit(vxvv=[R*u.pc, v_R*u.km/u.s, v_phi*u.km/u.s, z*u.pc,
                      v_z*u.km/u.s, phi*u.rad])

    # Integrate for 10 periods
    t = np.linspace(0, 10*P, 1000)
    orb.integrate(t, pot, method='dop853_c')

    ra = orb.rap(use_physical=False)
    ecc = orb.e()
    a = ra / (1 + ecc)

    return a * _pc
