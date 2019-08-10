import numpy as np
import astropy.units as u
from scipy import integrate
from scipy import optimize
from astropy import constants

_G = constants.G.to(u.pc * u.km ** 2 / u.s ** 2 / u.solMass).value


def gamma_bh_df(epsilon, m_bh, m_cl, s, gamma=1):
    """Evaluate the distribution function f(E) for a spherical gamma-family
    cluster potential with a central supermassive black hole.

    Parameters
    ----------
    epsilon : float
        Relative energy in km^2/s^2.
    m_bh : float
        Black hole mass in solar masses.
    m_cl : float
        Total cluster mass in solar masses.
    s : float
        Cluster scale radius in pc.
    gamma : float, optional
        The gamma power.

    Returns
    -------
    f : float
        The distribution function evaluated at epsilon, in s^3 / (pc km)^3
    """
    if gamma >= 2:
        raise ValueError("Values of gamma greater than 2 are unsupported")

    # Convert to dimensionless form
    m_total = m_bh + m_cl
    mu = m_bh / m_total
    E = epsilon / (_G * m_total / s)

    # Evaluate the integral
    integral = _df_integral(E, mu, gamma)

    # Return the df in physical units
    return integral / 8 ** 0.5 / np.pi ** 2 / (_G * m_total * s) ** 1.5


def _omega(psi, mu, gamma):
    """Mapping function from the cluster potential to the overall potential.

    Parameters
    ----------
    psi : float
        Dimensionless value of the cluster potential.
    mu : float
        Ratio of BH mass to total mass.
    gamma : float
        The gamma power.

    Returns
    -------
    omega : float
        Dimensionless value of the overall potential.
    """
    return psi + mu * (-1 - psi + (1 + (-2 + gamma) * psi) ** (1 / (-2 + gamma)))


def _d_omega(psi, mu, gamma):
    """First derivative of omega.

    Parameters
    ----------
    psi : float
        Dimensionless value of the cluster potential.
    mu : float
        Ratio of BH mass to total mass.
    gamma : float
        The gamma power.

    Returns
    -------
    d_omega : float
        d omega / d psi
    """
    return 1 + mu * (-1 + (1 + (-2 + gamma) * psi) ** (-1 + 1 / (-2 + gamma)))


def _d2_omega(psi, mu, gamma):
    """Second derivative of omega.

    Parameters
    ----------
    psi : float
        Dimensionless value of the cluster potential.
    mu : float
        Ratio of BH mass to total mass.
    gamma : float
        The gamma power.

    Returns
    -------
    d2_omega : float
        d^2 omega / d psi^2
    """
    return -((-3 + gamma) * mu * (1 + (-2 + gamma) * psi) ** (-2 + 1 / (-2 + gamma)))


def _omega_inverse(psi, mu, gamma):
    """Returns psi_star = omega^(-1)(psi)

    Parameters
    ----------
    psi : float
        Dimensionless value of the overall potential.
    mu : float
        Ratio of BH mass to total mass.
    gamma : float
        The gamma power.

    Returns
    -------
    psi_star : float
        psi_star such that omega(psi_star) = psi
    """
    return optimize.brentq(lambda psi_star: _omega(psi_star, mu, gamma) - psi, 0, 1 / (2 - gamma) - 1e-12)


def _d_nu(psi, gamma):
    """First derivative of the spatial stellar probability density.

    Parameters
    ----------
    psi : float
        Dimensionless value of the cluster potential.
    gamma : float
        The gamma power.

    Returns
    -------
    d_nu : float
        d nu / d psi
    """
    return -(((-3 + gamma) * (-1 + (1 + (-2 + gamma) * psi) ** (1 / (2 - gamma))) ** 3
              * (-gamma + (-4 + gamma) * (1 + (-2 + gamma) * psi) ** (1 / (2 - gamma))))
             / (4. * ((1 + (-2 + gamma) * psi) ** (1 / (2 - gamma))) ** gamma * (np.pi + np.pi * (-2 + gamma) * psi)))


def _d2_nu(psi, gamma):
    """Second derivative of the spatial stellar probability density.

    Parameters
    ----------
    psi : float
        Dimensionless value of the cluster potential.
    gamma : float
        The gamma power.

    Returns
    -------
    d2_nu : float
        d2 nu / d psi^2
    """
    return (((-3 + gamma) * (-gamma + 2 * (-1 + gamma) * (1 + (-2 + gamma) * psi) ** (1 / (2 - gamma))
                             + (-4 + gamma) / (1 + (-2 + gamma) * psi) ** (4 / (-2 + gamma))
                             - (2 * (-3 + gamma)) / (1 + (-2 + gamma) * psi) ** (3 / (-2 + gamma))))
            / (2. * np.pi * (1 + (-2 + gamma) * psi) ** 2 * ((1 + (-2 + gamma) * psi) ** (1 / (2 - gamma))) ** gamma))


def _integrand(psi, epsilon_star, mu, gamma):
    """Integrand of Eddington's formula.

    Parameters
    ----------
    psi : float
        Dimensionless value of the cluster potential.
    epsilon_star : float
        Cluster potential value at a given energy epsilon, omega^-1(epsilon).
    mu : float
        Ratio of BH mass to total mass.
    gamma : float
        The gamma power.

    Returns
    -------
    integrand : float
        The value of the integrand.
    """
    # omega and its derivatives
    omega_epsilon = _omega(epsilon_star, mu, gamma)
    omega = _omega(psi, mu, gamma)
    d_omega = _d_omega(psi, mu, gamma)
    d2_omega = _d2_omega(psi, mu, gamma)

    # nu and its derivatives
    d_nu = _d_nu(psi, gamma)
    d2_nu = _d2_nu(psi, gamma)

    # d/d_psi [(d_nu/d_psi) / (d_omega/d_psi)]
    d_nu_d_omega = (d2_nu * d_omega - d_nu * d2_omega) / d_omega ** 2

    return d_nu_d_omega / (omega_epsilon - omega) ** 0.5


def _df_integral(epsilon, mu, gamma):
    """Integral of Eddington's formula.

    Parameters
    ----------
    epsilon : float
        Dimensionless value of the overall potential.
    mu : float
        Ratio of BH mass to total mass.
    gamma : float
        The gamma power.

    Returns
    -------
    integral : float
        The value of the integral.
    """
    try:
        epsilon_star = _omega_inverse(epsilon, mu, gamma)
        integral = integrate.quad(lambda psi: _integrand(psi, epsilon_star, mu, gamma), 0, epsilon_star)[0]
    except (RuntimeError, ValueError, ZeroDivisionError):
        return np.nan
    return integral
