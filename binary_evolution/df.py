import numpy as np
import astropy.units as u
from scipy import integrate
from scipy import optimize
from astropy import constants

_G = constants.G.to(u.pc * u.km ** 2 / u.s ** 2 / u.solMass).value


class GammaWithBlackHoleDF:
    """
    A class representing the distribution function of a gamma-family star
    cluster with a central supermassive black hole, adapted from Baes, Dejonghe,
    and Buyle (2005) [arXiv:astro-ph/0411202].
    """

    def __init__(self, m_bh, m_cl, s, gamma=1):
        """Initialize this GammaWithBlackHoleDF.

        Parameters
        ----------
        m_bh : float
            Black hole mass in solar masses.
        m_cl : float
            Cluster mass in solar masses.
        s : float
            Cluster scale radius in pc.
        gamma : float, optional
            The gamma power.
        """
        if gamma > 2:
            raise ValueError("Values of gamma greater than 2 are unsupported")

        self._m_bh = m_bh
        self._m_cl = m_cl
        self._m_total = self._m_bh + self._m_cl
        self._mu = self._m_bh / self._m_total
        self._s = s
        self._gamma = gamma

    def __call__(self, epsilon, epsabs=1.49e-8, epsrel=1.49e-8, limit=50):
        """Evaluate the distribution function f(E) for this
        GammaWithBlackHoleDF. Equivalent to GammaWithBlackHoleDF.f.

        Parameters
        ----------
        epsilon : float
            Relative energy in km^2/s^2.
        epsabs : float or int, optional
            Absolute error tolerance for the integrator.
        epsrel : float or int, optional
            Relative error tolerance for the integrator.
        limit : float or int, optional
            An upper bound on the number of subintervals used in the adaptive
            algorithm of the integrator.

        Returns
        -------
        f : float
            The distribution function evaluated at epsilon, in s^3 / (pc km)^3
        """
        return self.f(epsilon, epsabs=epsabs, epsrel=epsrel, limit=limit)

    def f(self, epsilon, epsabs=1.49e-8, epsrel=1.49e-8, limit=50):
        """Evaluate the distribution function f(E) for this
        GammaWithBlackHoleDF.

        Parameters
        ----------
        epsilon : float
            Relative energy in km^2/s^2.
        epsabs : float or int, optional
            Absolute error tolerance for the integrator.
        epsrel : float or int, optional
            Relative error tolerance for the integrator.
        limit : float or int, optional
            An upper bound on the number of subintervals used in the adaptive
            algorithm of the integrator.

        Returns
        -------
        f : float
            The distribution function evaluated at epsilon, in s^3 / (pc km)^3
        """
        # Convert to dimensionless form
        E = epsilon / (_G * self._m_total / self._s)

        # Evaluate the integral
        integral = self._df_integral(E, epsabs=epsabs, epsrel=epsrel, limit=limit)

        # Return the df in physical units
        return integral / 8 ** 0.5 / np.pi ** 2 / (_G * self._m_total * self._s) ** 1.5

    def g(self, epsilon, epsabs=1.49e-8, epsrel=1.49e-8, limit=50):
        """Evaluate the density-of-states function g(E) for this
        GammaWithBlackHoleDF.

        Parameters
        ----------
        epsilon : float
            Relative energy in km^2/s^2.
        epsabs : float or int, optional
            Absolute error tolerance for the integrator.
        epsrel : float or int, optional
            Relative error tolerance for the integrator.
        limit : float or int, optional
            An upper bound on the number of subintervals used in the adaptive
            algorithm of the integrator.

        Returns
        -------
        g : float
            The density-of-states function evaluated at epsilon, in pc^3 km / s
        """
        # Convert to dimensionless form
        E = epsilon / (_G * self._m_total / self._s)

        # Evaluate the integral
        integral = self._dos_integral(E, epsabs=epsabs, epsrel=epsrel, limit=limit)

        # Return the dos in physical units
        return integral * (4 * np.pi) ** 2 * self._s ** 3 * (_G * self._m_total / self._s) ** 0.5

    def n(self, epsilon, epsabs=1.49e-8, epsrel=1.49e-8, limit=50):
        """Evaluate the differential energy distribution N(E) for this
        GammaWithBlackHoleDF.

        Parameters
        ----------
        epsilon : float
            Relative energy in km^2/s^2.
        epsabs : float or int, optional
            Absolute error tolerance for the integrator.
        epsrel : float or int, optional
            Relative error tolerance for the integrator.
        limit : float or int, optional
            An upper bound on the number of subintervals used in the adaptive
            algorithm of the integrator.

        Returns
        -------
        N : float
            The differential energy distribution evaluated at epsilon, in
            s^2 / km^2.
        """
        return (self.f(epsilon, epsabs=epsabs, epsrel=epsrel, limit=limit)
                * self.g(epsilon, epsabs=epsabs, epsrel=epsrel, limit=limit))

    def m_bh(self):
        """Return the mass of the central SMBH.

        Returns
        -------
        m_bh : float
            Black hole mass in solar masses
        """
        return self._m_bh

    def m_cl(self):
        """Return the mass of the nuclear star cluster

        Returns
        -------
        m_cl : float
            Cluster mass in solar masses
        """
        return self._m_cl

    def m_total(self):
        """Return the total mass

        Returns
        -------
        m_total : float
            Total mass in solar masses
        """
        return self._m_total

    def mu(self):
        """Return the ratio of the black hole mass to the total mass.

        Returns
        -------
        mu : float
            m_bh / m_total
        """
        return self._mu

    def s(self):
        """Return the scale radius.

        Returns
        -------
        s : float
            The scale radius in pc.
        """
        return self._s

    def gamma(self):
        """Return the gamma power.

        Returns
        -------
        gamma : float
            The value of gamma.
        """
        return self._gamma

    def _r(self, psi):
        """r as a function of psi.

        Parameters
        ----------
        psi : float
            Dimensionless value of the cluster potential.

        Returns
        -------
        r : float
            Value of r at psi.
        """
        return -1 + 1 / (1 - (1 + (-2 + self._gamma) * psi) ** (1 / (2 - self._gamma)))

    def _d_r(self, psi):
        """First derivative of r

        Parameters
        ----------
        psi : float
            Dimensionless value of the cluster potential.

        Returns
        -------
        r : float
            d r / d psi
        """
        return -((1 + (-2 + self._gamma) * psi) ** (-1 + 1 / (2 - self._gamma))
                 / (-1 + (1 + (-2 + self._gamma) * psi) ** (1 / (2 - self._gamma))) ** 2)

    def _omega(self, psi):
        """Mapping function from the cluster potential to the overall potential.

        Parameters
        ----------
        psi : float
            Dimensionless value of the cluster potential.

        Returns
        -------
        omega : float
            Dimensionless value of the overall potential.
        """
        return psi + self._mu * (-1 - psi + (1 + (-2 + self._gamma) * psi) ** (1 / (-2 + self._gamma)))

    def _d_omega(self, psi):
        """First derivative of omega.

        Parameters
        ----------
        psi : float
            Dimensionless value of the cluster potential.

        Returns
        -------
        d_omega : float
            d omega / d psi
        """
        return 1 + self._mu * (-1 + (1 + (-2 + self._gamma) * psi) ** (-1 + 1 / (-2 + self._gamma)))

    def _d2_omega(self, psi):
        """Second derivative of omega.

        Parameters
        ----------
        psi : float
            Dimensionless value of the cluster potential.

        Returns
        -------
        d2_omega : float
            d^2 omega / d psi^2
        """
        return -((-3 + self._gamma) * self._mu * (1 + (-2 + self._gamma) * psi) ** (-2 + 1 / (-2 + self._gamma)))

    def _omega_inverse(self, psi):
        """Returns psi_star = omega^(-1)(psi)

        Parameters
        ----------
        psi : float
            Dimensionless value of the overall potential.

        Returns
        -------
        psi_star : float
            psi_star such that omega(psi_star) = psi
        """
        return optimize.brentq(lambda psi_star: self._omega(psi_star) - psi, 0, 1 / (2 - self._gamma) - 1e-12)

    def _d_nu(self, psi):
        """First derivative of the spatial stellar probability density.

        Parameters
        ----------
        psi : float
            Dimensionless value of the cluster potential.

        Returns
        -------
        d_nu : float
            d nu / d psi
        """
        return -(((-3 + self._gamma) * (-1 + (1 + (-2 + self._gamma) * psi) ** (1 / (2 - self._gamma))) ** 3
                  * (-self._gamma + (-4 + self._gamma) * (1 + (-2 + self._gamma) * psi) ** (1 / (2 - self._gamma))))
                 / (4. * ((1 + (-2 + self._gamma) * psi) ** (1 / (2 - self._gamma)))
                    ** self._gamma * (np.pi + np.pi * (-2 + self._gamma) * psi)))

    def _d2_nu(self, psi):
        """Second derivative of the spatial stellar probability density.

        Parameters
        ----------
        psi : float
            Dimensionless value of the cluster potential.

        Returns
        -------
        d2_nu : float
            d2 nu / d psi^2
        """
        return (((-3 + self._gamma)
                 * (-self._gamma + 2 * (-1 + self._gamma) * (1 + (-2 + self._gamma) * psi) ** (1 / (2 - self._gamma))
                    + (-4 + self._gamma) / (1 + (-2 + self._gamma) * psi) ** (4 / (-2 + self._gamma))
                    - (2 * (-3 + self._gamma)) / (1 + (-2 + self._gamma) * psi) ** (3 / (-2 + self._gamma))))
                / (2. * np.pi * (1 + (-2 + self._gamma) * psi) ** 2
                   * ((1 + (-2 + self._gamma) * psi) ** (1 / (2 - self._gamma))) ** self._gamma))

    def _df_integrand(self, psi, epsilon_star):
        """Integrand of Eddington's formula.

        Parameters
        ----------
        psi : float
            Dimensionless value of the cluster potential.
        epsilon_star : float
            Cluster potential value at a given energy epsilon, omega^-1(epsilon).

        Returns
        -------
        integrand : float
            The value of the integrand.
        """
        # omega and its derivatives
        epsilon = self._omega(epsilon_star)
        omega = self._omega(psi)
        d_omega = self._d_omega(psi)
        d2_omega = self._d2_omega(psi)

        # nu and its derivatives
        d_nu = self._d_nu(psi)
        d2_nu = self._d2_nu(psi)

        # d/d_psi [(d_nu/d_psi) / (d_omega/d_psi)]
        d_nu_d_omega = (d2_nu * d_omega - d_nu * d2_omega) / d_omega ** 2

        return d_nu_d_omega / (epsilon - omega) ** 0.5

    def _df_integral(self, epsilon, epsabs=1.49e-8, epsrel=1.49e-8, limit=50):
        """Integral of Eddington's formula.

        Parameters
        ----------
        epsilon : float
            Dimensionless value of the overall potential.
        epsabs : float or int, optional
            Absolute error tolerance for the integrator.
        epsrel : float or int, optional
            Relative error tolerance for the integrator.
        limit : float or int, optional
            An upper bound on the number of subintervals used in the adaptive
            algorithm of the integrator.

        Returns
        -------
        integral : float
            The value of the integral.
        """
        try:
            epsilon_star = self._omega_inverse(epsilon)
            integral = integrate.quad(lambda psi: self._df_integrand(psi, epsilon_star),
                                      0, epsilon_star, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        except (RuntimeError, ValueError, ZeroDivisionError):
            return np.nan
        return integral

    def _dos_integrand(self, psi, epsilon_star):
        """Integrand for the density-of-states function.

        Parameters
        ----------
        psi : float
            Dimensionless value of the cluster potential.
        epsilon_star : float
            Cluster potential value at a given energy epsilon, omega^-1(epsilon).

        Returns
        -------
        integrand : float
            The value of the integrand.
        """
        epsilon = self._omega(epsilon_star)
        omega = self._omega(psi)

        r = self._r(psi)
        d_r = self._d_r(psi)

        return r ** 2 * d_r * (2 * (omega - epsilon)) ** 0.5

    def _dos_integral(self, epsilon, epsabs=1.49e-8, epsrel=1.49e-8, limit=50):
        """Integral for the density-of-states function.

        Parameters
        ----------
        epsilon : float
            Dimensionless value of the overall potential.
        epsabs : float or int, optional
            Absolute error tolerance for the integrator.
        epsrel : float or int, optional
            Relative error tolerance for the integrator.
        limit : float or int, optional
            An upper bound on the number of subintervals used in the adaptive
            algorithm of the integrator.

        Returns
        -------
        integral : float
            The value of the integral.
        """
        try:
            psi0 = 1 / (2 - self._gamma)
            epsilon_star = self._omega_inverse(epsilon)
            integral = integrate.quad(lambda psi: self._dos_integrand(psi, epsilon_star),
                                      psi0, epsilon_star, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
        except (RuntimeError, ValueError, ZeroDivisionError):
            return np.nan
        return integral
