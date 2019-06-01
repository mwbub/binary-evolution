import warnings
import numpy as np
import astropy.units as u
from astropy import constants
from galpy.orbit import Orbit
from galpy.potential import ttensor
from scipy.integrate import solve_ivp
from .vector_conversion import elements_to_vectors

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
            Argument of the pericentre in radians.
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

        # Result arrays
        self._t = None   # Time array
        self._e = None   # e vector array
        self._j = None   # j vector array
        self._r = None   # Position vector array
        self._v = None   # Velocity vector array

        # Check that e0 and j0 are valid
        if self._e0.shape != (3,) or self._j0.shape != (3,):
            raise ValueError("Orbital elements must be scalars, not arrays")

    def integrate(self, t, pot=None, func=None, alt_pot=None):
        """Integrate the orbit of this KeplerRing.

        Parameters
        ----------
        t : array_like
            Array of times at which to output, in years. Must be 1D and sorted.
        pot : galpy.potential.Potential or list of Potentials, optional
            A potential used to integrate the orbit. This potential's tidal
            tensor will be used to evolve the e and j vectors. If not provided,
            you must provide both a func and alt_pot parameter to integrate the
            e/j vectors and barycentre, respectively.
        func : callable, optional
            An additional term to add to the derivatives of the e and j vectors.
            The calling signature is func(t, e, j, r) where t is the time step,
            e and j are the eccentricity and dimensionless angular momentum
            vectors, and r is the position vector of the barycentre in Cartesian
            coordinates. The return value must be a tuple (de, dj), where de and
            dj are arrays of shape (3,) representing the derivatives of e and j.
        alt_pot : galpy.potential.Potential or list of Potentials, optional
            An additional potential used to integrate the barycentre position,
            but not to evolve the e and j vectors.

        Returns
        -------
        None
        """
        if pot is None and (func is None or alt_pot is None):
            raise KeplerRingError("Both func and alt_pot must be provided if "
                                  "pot is not provided")

        # Construct the potential to evolve the barycentre
        barycentre_pot = []
        if pot is not None:
            barycentre_pot.append(pot)
        if alt_pot is not None:
            barycentre_pot.append(alt_pot)

        # Integrate the barycentre
        orb = self._integrate_r(t, barycentre_pot)

        # Function to extract the r vector in Cartesian coordinates
        def r(time):
            x = orb.x(time*u.yr) * 1000
            y = orb.y(time*u.yr) * 1000
            z = orb.z(time.u.yr) * 1000
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

        self._integrate_ej(t, de_dj)

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
        if t is None:
            return self._e0

        if self._e is None:
            raise KeplerRingError("You must integrate this KeplerRing before "
                                  "evaluating e at a specific time step")

        t = np.array(t).flatten()
        result = self._e[[np.where(self._t == time)[0][0] for time in t]]

        if result.shape[0] == 1:
            return result[0]
        return result

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
        if t is None:
            return self._j0

        if self._j is None:
            raise KeplerRingError("You must integrate this KeplerRing before "
                                  "evaluating j at a specific time step")

        t = np.array(t).flatten()
        result = self._j[[np.where(self._t == time)[0][0] for time in t]]

        if result.shape[0] == 1:
            return result[0]
        return result

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
        if t is None:
            return self._r0

        if self._r is None:
            raise KeplerRingError("You must integrate this KeplerRing before "
                                  "evaluating r at a specific time step")

        t = np.array(t).flatten()
        result = self._r[[np.where(self._t == time)[0][0] for time in t]]

        if result.shape[0] == 1:
            return result[0]
        return result

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
        if t is None:
            return self._v0

        if self._v is None:
            raise KeplerRingError("You must integrate this KeplerRing before "
                                  "evaluating v at a specific time step")

        t = np.array(t).flatten()
        result = self._v[[np.where(self._t == time)[0][0] for time in t]]

        if result.shape[0] == 1:
            return result[0]
        return result

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

    def _integrate_ej(self, t, func):
        """Integrate the e and j vectors of this KeplerRing.

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

        Returns
        -------
        None
        """
        t = np.array(t)

        # Combine e/j into a single vector and solve the IVP
        ej0 = np.hstack((self._e0, self._j0))
        sol = solve_ivp(lambda time, x: np.hstack(func(time, x[:3], x[3:])),
                        (t[0], t[-1]), ej0, t_eval=t)

        # Save the results if the integration was successful
        if sol.success:
            self._e = sol.y[:3].T
            self._j = sol.y[3:].T
            self._t = t
        else:
            raise KeplerRingError("Integration of e and j vectors failed")

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
        j_cross_x = np.cross(j, [1, 0, 0])
        j_cross_y = np.cross(j, [0, 1, 0])
        j_cross_z = np.cross(j, [0, 0, 1])
        e_cross_x = np.cross(e, [1, 0, 0])
        e_cross_y = np.cross(e, [0, 1, 0])
        e_cross_z = np.cross(e, [0, 0, 1])

        # Stack the cross products into arrays for vectorized operations
        j_cross = np.vstack((j_cross_x, j_cross_y, j_cross_z))
        e_cross = np.vstack((e_cross_x, e_cross_y, e_cross_z))

        # Array of vectors of the form (n_beta dot j)(j cross n_alpha)
        j_j_cross = j[:, np.newaxis, np.newaxis] * j_cross
        e_e_cross = e[:, np.newaxis, np.newaxis] * e_cross
        j_e_cross = j[:, np.newaxis, np.newaxis] * e_cross
        e_j_cross = e[:, np.newaxis, np.newaxis] * j_cross

        # Calculate the terms inside the sum
        j_sum = tt[:, :, np.newaxis] * (j_j_cross - 5 * e_e_cross)
        e_sum = tt[:, :, np.newaxis] * (j_e_cross - 5 * e_j_cross)

        # Compute the sums and the trace term
        j_sum = np.sum(j_sum, (0, 1))
        e_sum = np.sum(e_sum, (0, 1))
        trace_term = np.trace(tt) * j_cross_e

        # Constant factor
        tau = self._a**1.5 / 2 / _G**0.5 / self._m**0.5

        # Derivatives
        dj = tau * j_sum
        de = tau * (trace_term + e_sum)

        return de, dj

    def _orthogonal_normal(self):
        """Sanity check to ensure that the e and j vectors are orthogonal and
        their norms sum to unit length.

        Returns
        -------
        min_dot : float
            Minimum dot product e dot j.
        max_dot : float
            Maximum dot product e dot j.
        min_norm : float
            Minimum of the norms (|e|^2 + |j|^2)^(1/2).
        max_norm : float
            Maximum of the norms (|e|^2 + |j|^2)^(1/2).
        """
        initial_norm = np.sum(self._e0**2 + self._j0**2)**0.5
        initial_dot = np.dot(self._e0, self._j0)

        if self._e is not None and self._j is not None:
            norms = np.sum(self._e**2 + self._j**2, axis=1)**0.5
            dots = np.array([np.dot(self._e[i], self._j[i]) for i in
                             range(len(self._e))])
            min_norm = np.min(norms)
            max_norm = np.max(norms)
            min_dot = np.min(dots)
            max_dot = np.max(dots)
        else:
            min_norm = initial_norm
            max_norm = initial_norm
            min_dot = initial_dot
            max_dot = initial_dot

        return min_dot, max_dot, min_norm, max_norm


class KeplerRingError(Exception):
    pass


class KeplerRingWarning(Warning):
    pass
