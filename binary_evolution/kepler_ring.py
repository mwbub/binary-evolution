import warnings
import numpy as np
import astropy.units as u
from astropy import constants
from galpy.orbit import Orbit
from galpy.potential import ttensor
from scipy.integrate import solve_ivp
from .vector_conversion import elements_to_vectors, vectors_to_elements

_G = constants.G.to(u.pc**3/u.solMass/u.yr**2).value


class KeplerRing:
    """
    A class used to evolve a Keplerian ring using vectorial formalism.
    """

    def __init__(self, ecc, inc, long_asc, arg_peri, r, v, m=1, a=1):
        """Initialize a Keplerian ring.

        Parameters
        ----------
        ecc : float
            Eccentricity. Must be between 0 and 1.
        inc : float
            Inclination relative to the x-y plane in radians.
        long_asc : float
            Longitude of the ascending node in radians.
        arg_peri : float
            Argument of pericentre in radians.
        r : array_like
            Initial position of the barycentre in Galactocentric cylindrical
            coordinates, of the form [R, z, phi] in [pc, pc, rad].
        v : array_like
            Initial velocity of the barycentre in Galactocentric cylindrical
            coordinates, of the form [v_R, v_z, v_phi] in km/s.
        a : float, optional
            Semi-major axis of the ring in AU.
        m : float, optional
            Total mass of the ring in solar masses.
        """
        # Initial conditions
        self._e0, self._j0 = elements_to_vectors(ecc, inc, long_asc, arg_peri)
        self._r0 = np.array(r)
        self._v0 = np.array(v)
        self._a = (a*u.au).to(u.pc).value
        self._m = m

        # Constant factor for integration
        self._tau = self._a ** 1.5 / 2 / _G ** 0.5 / self._m ** 0.5

        # Result arrays
        self._t = None   # Time array
        self._e = None   # e vector array
        self._j = None   # j vector array
        self._r = None   # Position vector array
        self._v = None   # Velocity vector array

        # Check that e0 and j0 are valid
        if self._e0.shape != (3,) or self._j0.shape != (3,):
            raise ValueError("Orbital elements must be scalars, not arrays")

    def integrate(self, t, pot=None, func=None, r_pot=None, rtol=1e-6,
                  atol=1e-12):
        """Integrate the orbit of this KeplerRing.

        Parameters
        ----------
        t : array_like
            Array of times at which to output, in years. Must be 1D and sorted.
        pot : galpy.potential.Potential or list of Potentials, optional
            A potential used to integrate the orbit. This potential's tidal
            tensor will be used to evolve the e and j vectors. If not provided,
            you must provide both a func and r_pot parameter to integrate the
            e/j vectors and barycentre, respectively.
        func : callable, optional
            An additional term to add to the derivatives of the e and j vectors.
            The calling signature is func(t, e, j, r) where t is the time step,
            e and j are the eccentricity and dimensionless angular momentum
            vectors, and r is the position vector of the barycentre in Cartesian
            coordinates. The return value must be a tuple (de, dj), where de and
            dj are arrays of shape (3,) representing the derivatives of e and j.
        r_pot : galpy.potential.Potential or list of Potentials, optional
            An additional potential used to integrate the barycentre position,
            but not to evolve the e and j vectors. This potential will be summed
            with pot to integrate the r vector.
        rtol, atol : float or array_like, optional
            Relative and absolute error tolerances for the solver. Here, rtol
            controls the number of correct digits, while atol controls the
            threshold below which the precision of a component of e or j is no
            longer guaranteed. For more details, see the documentation of the
            scipy.integrate.solve_ivp function.

        Returns
        -------
        None
        """
        if pot is None and (func is None or r_pot is None):
            raise KeplerRingError("Both func and r_pot must be provided if "
                                  "pot is not provided")

        # Construct the potential to evolve the barycentre
        barycentre_pot = []
        if pot is not None:
            barycentre_pot.append(pot)
        if r_pot is not None:
            barycentre_pot.append(r_pot)

        # Integrate the barycentre
        orb = self._integrate_r(t, barycentre_pot)

        # Function to extract the r vector in Cartesian coordinates
        def r(time):
            x = orb.x(time*u.yr) * 1000
            y = orb.y(time*u.yr) * 1000
            z = orb.z(time*u.yr) * 1000
            return np.array([x, y, z])

        # Function to extract the r vector in cylindrical coordinates
        def r_cyl(time):
            R = orb.R(time*u.yr) * 1000
            z = orb.z(time*u.yr) * 1000
            phi = orb.phi(time*u.yr)
            return np.array([R, z, phi])

        # Combined derivative function
        if pot is not None and func is not None:
            def de_dj(time, e, j):
                de, dj = self._tidal_derivatives(pot, time, e, j, r_cyl(time))
                de_alt, dj_alt = func(time, e, j, r(time))
                return de + de_alt, dj + dj_alt
        elif pot is not None:
            def de_dj(time, e, j):
                return self._tidal_derivatives(pot, time, e, j, r_cyl(time))
        elif func is not None:
            def de_dj(time, e, j):
                return func(time, e, j, r(time))
        else:
            raise KeplerRingError("Both pot and func are unprovided")

        self._integrate_ej(t, de_dj, rtol=rtol, atol=atol)

    def e(self, t=None):
        """Return the e vector at a specified time.

        Parameters
        ----------
        t : array_like, optional
            A time or array of times at which to retrieve e. All times must
            be contained within the KeplerRing.t() array.

        Returns
        -------
        e : ndarray
            e vector at the specified time steps.
        """
        return self._params(t=t)[0]

    def j(self, t=None):
        """Return the j vector at a specified time.

        Parameters
        ----------
        t : array_like, optional
            A time or array of times at which to retrieve j. All times must
            be contained within the KeplerRing.t() array.

        Returns
        -------
        j : ndarray
            j vector at the specified time steps.
        """
        return self._params(t=t)[1]

    def r(self, t=None):
        """Return the position vector at a specified time.

        Parameters
        ----------
        t : array_like, optional
            A time or array of times at which to retrieve r. All times must
            be contained within the KeplerRing.t() array.

        Returns
        -------
        r : ndarray
            Position vector at the specified time steps. Has the form
            [R, z, phi] in [pc, pc, rad].
        """
        return self._params(t=t)[2]

    def v(self, t=None):
        """Return the velocity vector at a specified time.

        Parameters
        ----------
        t : array_like, optional
            A time or array of times at which to retrieve v. All times must
            be contained within the KeplerRing.t() array.

        Returns
        -------
        v : ndarray
            Velocity vector at the specified time steps. Has the form
            [v_R, v_z, v_phi] in km/s.
        """
        return self._params(t=t)[3]

    def ecc(self, t=None):
        """Return the eccentricity at a specified time.

        Parameters
        ----------
        t : array_like, optional
            A time or array of times at which to retrieve the eccentricity. All
            times must be contained within the KeplerRing.t() array.

        Returns
        -------
        ecc : float or ndarray
            The eccentricity at the specified time steps.
        """
        return self._params(t=t)[4]

    def inc(self, t=None):
        """Return the inclination at a specified time.

        Parameters
        ----------
        t : array_like, optional
            A time or array of times at which to retrieve the inclination. All
            times must be contained within the KeplerRing.t() array.

        Returns
        -------
        inc : float or ndarray
            The inclination in radians at the specified time steps.
        """
        return self._params(t=t)[5]

    def long_asc(self, t=None):
        """Return the longitude of the ascending node at a specified time.

        Parameters
        ----------
        t : array_like, optional
            A time or array of times at which to retrieve the longitude of the
            ascending node. All times must be contained within the
            KeplerRing.t() array.

        Returns
        -------
        long_asc : float or ndarray
            Longitude of the ascending node in radians at the specified time
            step.
        """
        return self._params(t=t)[6]

    def arg_peri(self, t=None):
        """Return the argument of pericentre at a specified time step.

        Parameters
        ----------
        t : array_like, optional
            A time or array of times at which to retrieve the argument of the
            pericentre. All times must be contained within the KeplerRing.t()
            array.

        Returns
        -------
        arg_peri : float or ndarray
            Argument of pericentre in radians at the specified time step.
        """
        return self._params(t=t)[7]

    def t(self):
        """Return the time array used to integrate this KeplerRing.

        Returns
        -------
        t : ndarray
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
        m : float
            The mass in solar masses.
        """
        return self._m

    def a(self, pc=False):
        """Return the semi-major axis of this KeplerRing.

        Parameters
        ----------
        pc : bool, optional
            If True, return the semi-major axis in pc. Otherwise return in AU.

        Returns
        -------
        a : float
            The semi-major axis.
        """
        if pc:
            return self._a
        return (self._a*u.pc).to(u.au).value

    def _integrate_ej(self, t, func, rtol=1e-6, atol=1e-12):
        """Integrate the e and j vectors of this KeplerRing. Uses an explicit
        Runge-Kutta method of order 5(4) from scipy's solve_ivp.

        Parameters
        ----------
        t : array_like
            Array of times at which to output, in years. Must be 1D and sorted.
        func : callable
            The derivative function of e and j. The calling signature is
            func(t, e, j), where t is the time step, and e and j are the
            eccentricity and dimensionless angular momentum vectors. The return
            value must be a tuple (de, dj), where de and dj are arrays of shape
            (3,) representing the derivatives of the e and j vectors.
        rtol, atol : float or array_like, optional
            Relative and absolute error tolerances for the solver. Here, rtol
            controls the number of correct digits, while atol controls the
            threshold below which the precision of a component of e or j is no
            longer guaranteed. For more details, see the documentation of the
            scipy.integrate.solve_ivp function.

        Returns
        -------
        None
        """
        t = np.array(t)

        # Combine e/j into a single vector and solve the IVP
        ej0 = np.hstack((self._e0, self._j0))
        sol = solve_ivp(lambda time, x: np.hstack(func(time, x[:3], x[3:])),
                        (t[0], t[-1]), ej0, t_eval=t, method='RK45', rtol=rtol,
                        atol=atol)

        # Save the results if the integration was successful
        if sol.success:
            self._e = sol.y[:3].T
            self._j = sol.y[3:].T
            self._t = t
        else:
            raise KeplerRingError("Integration of e and j vectors failed")

        # Sanity checks
        dot_err, norm_err = self._error()

        if dot_err > rtol * 10:
            msg = ("The error in the orthogonality of e and j is {:.1e}, which "
                   "exceeds the provided rtol of {:.1e}").format(dot_err, rtol)
            warnings.warn(msg, KeplerRingWarning)

        if norm_err > rtol * 10:
            msg = ("The error in the norm of e and j is {:.1e}, which exceeds "
                   "the provided rtol of {:.1e}").format(norm_err, rtol)
            warnings.warn(msg, KeplerRingWarning)

    def _integrate_r(self, t, pot):
        """Integrate the position vector of the barycentre of this KeplerRing.

        Parameters
        ----------
        t : array_like
            Array of times at which to output, in years. Must be 1D and sorted.
        pot : galpy.potential.Potential or list of Potentials
            A potential used to integrate the orbit.

        Returns
        -------
        orb : galpy.orbit.Orbit
            An Orbit instance containing the integrated orbit.
        """
        t = np.array(t)

        # Set up the Orbit instance
        R, z, phi = self._r0
        v_R, v_z, v_phi = self._v0
        orb = Orbit(vxvv=[R*u.pc, v_R*u.km/u.s, v_phi*u.km/u.s, z*u.pc,
                          v_z*u.km/u.s, phi*u.rad])

        # Integrate the orbit
        orb.integrate(t*u.yr, pot)

        # Extract the coordinates and convert to proper units
        R = orb.R(t*u.yr) * 1000
        z = orb.z(t*u.yr) * 1000
        phi = orb.phi(t*u.yr)
        v_R = orb.vR(t*u.yr)
        v_z = orb.vz(t*u.yr)
        v_phi = orb.vT(t*u.yr)

        # Save the results at each time step
        self._r = np.vstack((R, z, phi)).T
        self._v = np.vstack((v_R, v_z, v_phi)).T
        self._t = t

        return orb

    def _tidal_derivatives(self, pot, t, e, j, r):
        """Compute the derivatives of the e and j vector due to a tidal field.

        Parameters
        ----------
        pot : galpy.potential.Potential or list of Potentials
            The potential which originates the tidal field.
        t : float
            The time of evaluation in years.
        e : ndarray
            The eccentricity vector, of the form [ex, ey, ez].
        j : ndarray
            The dimensionless angular momentum vector, of the form [jx, jy, jz].
        r : ndarray
            Position vector of the barycentre in Galactocentric cylindrical
            coordinates, of the form [R, z, phi] in [pc, pc, rad].

        Returns
        -------
        de : ndarray
            An array of shape (3,) representing the derivative of e.
        dj : ndarray
            An array of shape (3,) representing the derivative of j.
        """
        # Extract the coordinates
        R, z, phi = r

        # Calculate the tidal tensor and convert from Gyr^-2 to yr^-2
        tt = -ttensor(pot, R*u.pc, z*u.pc, phi=phi, t=t*u.yr, vo=220, ro=8)
        tt /= (10**9)**2

        # Pre-compute the cross products
        j_cross_e = np.cross(j, e)
        j_cross_ni = np.cross(j, np.identity(3))
        e_cross_ni = np.cross(e, np.identity(3))

        # Array of vectors of the form (n_beta dot j)(j cross n_alpha), etc.
        jj = j[:, np.newaxis, np.newaxis] * j_cross_ni
        ee = e[:, np.newaxis, np.newaxis] * e_cross_ni
        je = j[:, np.newaxis, np.newaxis] * e_cross_ni
        ej = e[:, np.newaxis, np.newaxis] * j_cross_ni

        # Calculate the terms inside the sums
        j_sum = tt[:, :, np.newaxis] * (jj - 5 * ee)
        e_sum = tt[:, :, np.newaxis] * (je - 5 * ej)

        # Compute the sums and the trace term
        j_sum = np.sum(j_sum, (0, 1))
        e_sum = np.sum(e_sum, (0, 1))
        trace_term = np.trace(tt) * j_cross_e

        # Derivatives
        dj = self._tau * j_sum
        de = self._tau * (trace_term + e_sum)

        return de, dj

    def _error(self):
        """Sanity check to ensure that the e and j vectors are orthogonal and
        their norms sum to unit length.

        Returns
        -------
        dot_err : float
            Maximum error of the dot product (e dot j) from 0.
        norm_err : float
            Maximum error of the norm (|e|^2 + |j|^2)^(1/2) from 1.
        """
        initial_norm = np.sum(self._e0**2 + self._j0**2)**0.5
        initial_dot = np.dot(self._e0, self._j0)

        if self._e is not None and self._j is not None:
            norms = np.sum(self._e**2 + self._j**2, axis=1)**0.5
            dots = np.sum(self._e * self._j, axis=-1)
            dot_err = np.nanmax(np.abs(dots))
            norm_err = np.nanmax(np.abs(1 - norms))
        else:
            dot_err = np.abs(initial_dot)
            norm_err = np.abs(1 - initial_norm)

        return dot_err, norm_err

    def _params(self, t=None):
        """Return a tuple of all time-dependent parameters at a specified time.

        Parameters
        ----------
        t : array_like, optional
            A time or array of times at which to retrieve the parameters. All
            times must be contained within the KeplerRing.t() array.

        Returns
        -------
        e : ndarray
            e vector at the specified time steps.
        j : ndarray
            j vector at the specified time steps.
        r : ndarray
            Position vector at the specified time steps. Has the form
            [R, z, phi] in [pc, pc, rad].
        v : ndarray
            Velocity vector at the specified time steps. Has the form
            [v_R, v_z, v_phi] in km/s.
        ecc : float or ndarray
            Eccentricity at the specified time steps.
        inc : float or ndarray
            Inclination at the specified time steps.
        long_asc : float or ndarray
            Longitude of the ascending node at the specified time steps.
        arg_peri : float or ndarray
            Argument of pericentre at the specified time steps.
        """
        if t is None:
            elements = vectors_to_elements(self._e0, self._j0)
            return (self._e0, self._j0, self._r0, self._v0) + elements

        if self._t is None:
            raise KeplerRingError("You must integrate this KeplerRing before "
                                  "evaluating it at a specific time step")

        t = np.array(t).flatten()
        indices = np.hstack([np.where(self._t == time)[0] for time in t])

        e = self._e[indices]
        j = self._j[indices]
        r = self._r[indices]
        v = self._v[indices]

        if indices.size == 1:
            e = e[0]
            j = j[0]
            r = r[0]
            v = v[0]

        return (e, j, r, v) + vectors_to_elements(e, j)


class KeplerRingError(Exception):
    pass


class KeplerRingWarning(Warning):
    pass
