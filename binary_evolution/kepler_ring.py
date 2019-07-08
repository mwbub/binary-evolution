# Standard library imports
import warnings

# Third party imports
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy import constants
from galpy.orbit import Orbit
from galpy.potential import vcirc
from galpy.actionAngle import UnboundError
from galpy.util.bovy_conversion import time_in_Gyr
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline

# Local imports
from .vector_conversion import elements_to_vectors, vectors_to_elements
from .tidal_tensor import TidalTensor

# Physical constants
_G = constants.G.to(u.pc**3/u.solMass/u.yr**2).value
_c = constants.c.to(u.pc/u.yr).value

# Factors for conversion into galpy internal units
_pc = 1 / 8000
_yr = 1 / time_in_Gyr(220, 8) / 1e+9


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
        self.set_elements(ecc, inc, long_asc, arg_peri, m=m, a=a)
        self._r0 = np.array(r)
        self._v0 = np.array(v)

        # Constant factor for integration
        self._tau = self._a ** 1.5 / 2 / _G ** 0.5 / self._m ** 0.5

        # Result arrays
        self._t = None   # Time array
        self._e = None   # e vector array
        self._j = None   # j vector array
        self._r = None   # Position vector array
        self._v = None   # Velocity vector array

        # List of splines to interpolate the integrated parameters
        self._interpolatedInner = None
        self._interpolatedOuter = None

        # Check that e0 and j0 are valid
        if self._e0.shape != (3,) or self._j0.shape != (3,):
            raise ValueError("Orbital elements must be scalars, not arrays")

    def set_elements(self, ecc, inc, long_asc, arg_peri, a=None, m=None):
        """Set the orbital elements of this KeplerRing.

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
        a : float, optional
            Semi-major axis of the ring in AU.
        m : float, optional
            Total mass of the ring in solar masses.

        Returns
        -------
        None
        """
        if a is not None:
            self._a = (a*u.au).to(u.pc).value
        if m is not None:
            self._m = m
        self._e0, self._j0 = elements_to_vectors(ecc, inc, long_asc, arg_peri)

        # Reset the result arrays
        self._t = None
        self._e = None
        self._j = None

    def integrate(self, t, pot=None, func=None, r_pot=None, rtol=1e-9,
                  atol=1e-12, r_method='dop853_c', ej_method='LSODA',
                  reintegrate=True, include_relativity=False):
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
        r_method : str, optional
            Method used to integrate the barycentre position. See the
            documentation for galpy.orbit.Orbit.integrate for available options.
        ej_method : str, optional
            Integration method for evolving the e and j vectors. See the
            documentation for scipy.integrate.solve_ivp for available options.
        reintegrate : boolean, optional
            If False, will attempt to re-use a previously calculated barycentre
            orbit rather than reintegrating from scratch. Otherwise, any
            previous integration results will be discarded.
        include_relativity : boolean, optional
            If True, will include the relativistic precession of the e vector.

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
        if reintegrate or self._interpolatedOuter is None:
            self._integrate_r(t, barycentre_pot, method=r_method)

        x_interpolated = self._interpolatedOuter['x']
        y_interpolated = self._interpolatedOuter['y']
        z_interpolated = self._interpolatedOuter['z']

        # Function to extract the r vector in Cartesian coordinates
        def r(time):
            x = x_interpolated(time)
            y = y_interpolated(time)
            z = z_interpolated(time)
            return np.array([x, y, z])

        # List of derivative functions to sum together
        funcs = []
        if pot is not None:
            ttensor = TidalTensor(pot)
            funcs.append(lambda *args: self._tidal_derivatives(ttensor, *args))
        if func is not None:
            funcs.append(func)
        if include_relativity:
            funcs.append(lambda *args: (self._gr_precession(*args[1:3]), 0))

        # Combined derivative function
        def derivatives(time, e, j):
            r_vec = r(time)
            return np.sum([f(time, e, j, r_vec) for f in funcs], axis=0)

        self._integrate_ej(t, derivatives, rtol=rtol, atol=atol,
                           method=ej_method)

    def e(self, t=None):
        """Return the e vector at a specified time.

        Parameters
        ----------
        t : array_like, optional
            A time or array of times at which to retrieve e.

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
            A time or array of times at which to retrieve j.

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
            A time or array of times at which to retrieve r.

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
            A time or array of times at which to retrieve v.

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
            A time or array of times at which to retrieve the eccentricity.

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
            A time or array of times at which to retrieve the inclination.

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
            ascending node.

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
            pericentre.

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

    def save(self, filename, t=None):
        """Save the orbit of this KeplerRing in a .fits file.

        Parameters
        ----------
        filename : str
            The filename of the output .fits archive.
        t : array_like, optional
            Array of time steps at which to save.

        Returns
        -------
        None
        """
        if self._t is None:
            raise KeplerRingError("Use KeplerRing.integrate() before using"
                                  "KeplerRing.save()")

        if filename.lower()[-5:] != ".fits":
            filename = filename + ".fits"

        if t is None:
            t = self._t

        r, v, ecc, inc, long_asc, arg_peri = self._params(t)[2:]

        if r.shape == (3,):
            raise KeplerRingError("t must contain at least 2 time steps")

        hdu = fits.BinTableHDU.from_columns([
            fits.Column(name='t', format='D', array=t),
            fits.Column(name='ecc', format='D', array=ecc),
            fits.Column(name='inc', format='D', array=inc),
            fits.Column(name='long_asc', format='D', array=long_asc),
            fits.Column(name='arg_peri', format='D', array=arg_peri),
            fits.Column(name='R', format='D', array=r[:, 0]),
            fits.Column(name='z', format='D', array=r[:, 1]),
            fits.Column(name='phi', format='D', array=r[:, 2]),
            fits.Column(name='v_R', format='D', array=v[:, 0]),
            fits.Column(name='v_z', format='D', array=v[:, 1]),
            fits.Column(name='v_phi', format='D', array=v[:, 2])
        ])

        hdu.header.set('M', self.m())
        hdu.header.set('A', self.a())
        hdu.writeto(filename)

    def gamma(self, pot, method='dop853_c', num_periods=200):
        """Calculate the gamma constant for this KeplerRing in a given
        potential, which is related to the maximum eccentricity.

        Parameters
        ----------
        pot : galpy.potential.Potential or list of Potentials
            The potential used to integrate this KeplerRing.
        method : str, optional
            Method used to integrate the barycentre position. See the
            documentation for galpy.orbit.Orbit.integrate for available options.
        num_periods : int, optional
            The approximate number of azimuthal periods over which to average.

        Returns
        -------
        gamma : float
            The gamma constant.
        """
        txx, tzz = self._ttensor_mean(pot, num_periods=num_periods,
                                      method=method)
        return (tzz - txx) / 3 / (tzz + txx)

    def e_max(self, pot, method='dop853_c'):
        """Calculate the predicted maximum eccentricity achieved by this
        KeplerRing in its Lidov-Kozai cycles, assuming a doubly-averaged
        potential

        Parameters
        ----------
        pot : galpy.potential.Potential or list of Potentials
            The potential used to integrate this KeplerRing.
        method : str, optional
            Method used to integrate the barycentre position. See the
            documentation for galpy.orbit.Orbit.integrate for available options.

        Returns
        -------
        e_max : The predicted maximum eccentricity.
        """
        gamma = self.gamma(pot, method=method)
        return (1 - 10 * gamma * np.cos(self.inc())**2 / (1 + 5 * gamma))**0.5

    def tau_nodal(self, pot, point_mass, method='dop853_c', num_periods=200):
        """Return the timescale of the nodal precession of this KeplerRing's
        outer orbit due to a cluster potential.

        Parameters
        ----------
        pot : galpy.potential.Potential or list of Potentials
            The cluster potential.
        point_mass : PointMass
            A point mass representing a black hole at the origin.
        method : str, optional
            Method used to integrate the barycentre position. See the
            documentation for galpy.orbit.Orbit.integrate for available options.
        num_periods : int, optional
            The approximate number of azimuthal periods over which to average.

        Returns
        -------
        tau_nodal : float
            The nodal precession timescale in years.
        """
        r_pot = point_mass.potential()
        txx, tzz = self._ttensor_mean(pot, r_pot=r_pot, num_periods=num_periods,
                                      method=method)
        r_mag = np.sum(self._r0[:2]**2)**0.5
        return 2 * (_G * point_mass.m())**0.5 / (r_mag**1.5 * np.abs(tzz - txx))

    def epsilon(self, pot, point_mass, method='dop853_c'):
        """Return the ratio between the timescale of the Lidov-Kozai cycles and
        the timescale of the nodal precession of this KeplerRing.

        Parameters
        ----------
        pot : galpy.potential.Potential or list of Potentials
            The cluster potential.
        point_mass : PointMass
            A point mass representing a black hole at the origin.
        method : str, optional
            Method used to integrate the barycentre position. See the
            documentation for galpy.orbit.Orbit.integrate for available options.

        Returns
        -------
        epsilon : float
            The ratio tau_lk / tau_nodal, where tau_lk is the timescale of the
            Lidov-Kozai cycles, and tau_nodal is the timescale of the nodal
            precession.
        """
        tau_lk = point_mass.tau(self)
        tau_nodal = self.tau_nodal(pot, point_mass, method=method)
        return tau_lk / tau_nodal

    def inc_out(self):
        """Return the initial inclination of the outer (barycentre) orbit with
        respect to the x-y plane.

        Returns
        -------
        inc_out : float
            The inclination in radians.
        """
        j_hat_out = self._j_hat_out()
        return np.arccos(np.dot(j_hat_out, [0, 0, 1]))

    def inc_in_out(self):
        """Return the inclination of the inner (binary) orbit with respect to
        the orbital plane of the outer (barycentre) orbit.

        Returns
        -------
        inc_in_out : float
            The inclination in radians
        """
        j_hat_in = self._j0 / np.linalg.norm(self._j0)
        j_hat_out = self._j_hat_out()
        return np.arccos(np.dot(j_hat_in, j_hat_out))

    def _j_hat_out(self):
        """Return the initial unit vector of the outer orbit angular momentum.

        Returns
        -------
        j_hat_out : ndarray
            An array of shape (3,) giving the unit vector in Cartesian
            coordinates at time 0.
        """
        R, z, phi = self._r0
        v_R, v_z, v_phi = self._v0

        x = R * np.cos(phi)
        y = R * np.sin(phi)
        v_x = v_R * np.cos(phi) - v_phi * np.sin(phi)
        v_y = v_R * np.sin(phi) + v_phi * np.cos(phi)

        J = np.cross([x, y, z], [v_x, v_y, v_z])
        return J / np.linalg.norm(J)

    def _integrate_ej(self, t, func, rtol=1e-9, atol=1e-12, method='LSODA'):
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
        rtol, atol : float or array_like, optional
            Relative and absolute error tolerances for the solver. Here, rtol
            controls the number of correct digits, while atol controls the
            threshold below which the precision of a component of e or j is no
            longer guaranteed. For more details, see the documentation of the
            scipy.integrate.solve_ivp function.
        method : str, optional
            Integration method for evolving the e and j vectors. See the
            documentation for scipy.integrate.solve_ivp for available options.

        Returns
        -------
        None
        """
        t = np.array(t)

        # Combine e/j into a single vector and solve the IVP
        ej0 = np.hstack((self._e0, self._j0))
        sol = solve_ivp(lambda time, x: np.hstack(func(time, x[:3], x[3:])),
                        (t[0], t[-1]), ej0, t_eval=t, method=method, rtol=rtol,
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

        self._setup_inner_interpolation()

    def _integrate_r(self, t, pot, method='dop853_c'):
        """Integrate the position vector of the barycentre of this KeplerRing.

        Parameters
        ----------
        t : array_like
            Array of times at which to output, in years. Must be 1D and sorted.
        pot : galpy.potential.Potential or list of Potentials
            A potential used to integrate the orbit.
        method : str, optional
            Method used to integrate the barycentre position. See the
            documentation for galpy.orbit.Orbit.integrate for available options.

        Returns
        -------
        None
        """
        t = np.array(t)

        orb = self._get_orbit()

        # Integrate the orbit
        orb.integrate(t*u.yr, pot, method=method)

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

        self._setup_outer_interpolation()

    def _tidal_derivatives(self, ttensor, t, e, j, r):
        """Compute the derivatives of the e and j vector due to a tidal field.

        Parameters
        ----------
        ttensor : TidalTensor
            TidalTensor instance containing the desired potential.
        t : float
            The time of evaluation in years.
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
        # Calculate the tidal tensor
        x, y, z = r
        tt = ttensor(x, y, z, t=t)

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

    def _gr_precession(self, e, j):
        """Compute the derivative of e due to relativistic precession.

        Parameters
        ----------
        e : ndarray
            The eccentricity vector, of the form [ex, ey, ez].
        j : ndarray
            The dimensionless angular momentum vector, of the form [jx, jy, jz].

        Returns
        -------
        de : ndarray
            An array of shape (3,) representing the derivative of e.
        """
        ecc = np.linalg.norm(e)
        j_cross_e = np.cross(j, e)
        tau = 3 * (_G * self._m)**1.5 / self._a**2.5 / _c**2 / (1 - ecc**2)**1.5
        return tau * j_cross_e

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

        # Inner binary parameters
        e = [self._interpolatedInner[k](t) for k in ('ex', 'ey', 'ez')]
        j = [self._interpolatedInner[k](t) for k in ('jx', 'jy', 'jz')]
        e = np.stack(e, axis=-1)
        j = np.stack(j, axis=-1)

        # Outer orbit parameters
        x, y, z = [self._interpolatedOuter[k](t) for k in ('x', 'y', 'z')]
        v = [self._interpolatedOuter[k](t) for k in ('v_R', 'v_z', 'v_phi')]
        R = (x**2 + y**2)**0.5
        phi = np.arctan2(y, x)
        r = np.stack((R, z, phi), axis=-1)
        v = np.stack(v, axis=-1)

        return (e, j, r, v) + vectors_to_elements(e, j)

    def _get_orbit(self):
        """Return a galpy Orbit using the initial conditions of this KeplerRing.

        Returns
        -------
        orb : galpy.orbit.Orbit
            An orbit containing the initial conditions of this KeplerRing.
        """
        R, z, phi = self._r0
        v_R, v_z, v_phi = self._v0
        orb = Orbit(vxvv=[R*u.pc, v_R*u.km/u.s, v_phi*u.km/u.s, z*u.pc,
                          v_z*u.km/u.s, phi*u.rad])
        return orb

    def _ttensor_mean(self, pot, r_pot=None, method='dop853_c',
                      num_periods=200):
        """Calculate the average tidal tensor of a potential over many orbits.

        Parameters
        ----------
        pot : galpy.potential.Potential or list of Potentials
            The potential used to evaluate the tidal tensor.
        r_pot : galpy.potential.Potential or list of Potentials.
            An additional potential to be summed with pot for the purpose of
            integrating the barycentre of this KeplerRing.
        method : str, optional
            Method used to integrate the barycentre position. See the
            documentation for galpy.orbit.Orbit.integrate for available options.
        num_periods : int, optional
            The approximate number of azimuthal periods over which to average.

        Returns
        -------
        txx : float
            Average xx component of the tidal tensor in yr^-2.
        tzz : float
            Average zz component of the tidal tensor in yr^-2.
        """
        if r_pot is None:
            barycentre_pot = pot
        else:
            barycentre_pot = [pot, r_pot]

        # Set up the orbit
        orb = self._get_orbit()

        # Calculate the orbital period, and assume circular if this fails
        try:
            P = orb.Tp(barycentre_pot, use_physical=False)
            if np.isnan(P):
                raise ValueError
        except (ValueError, ZeroDivisionError, NotImplementedError, TypeError,
                UnboundError):
            msg = ("Calculation of the azimuthal period failed. Assuming a "
                   "circular orbital period instead")
            warnings.warn(msg, KeplerRingWarning)
            orb_R = orb.R(use_physical=False)
            phi = orb.phi()
            vc = vcirc(barycentre_pot, orb_R, phi=phi, use_physical=False)
            P = orb_R * 2 * np.pi / vc

        # Integrate for 200 azimuthal periods
        t = np.linspace(0, P*num_periods, num_periods*20)
        orb.integrate(t, barycentre_pot, method=method)

        # Extract the coordinates from the orbit
        Rs = orb.R(t, use_physical=False) / _pc
        zs = orb.z(t, use_physical=False) / _pc
        phis = orb.phi(t)

        # Convert to Cartesian coordinates
        xs = Rs * np.cos(phis)
        ys = Rs * np.sin(phis)

        # Calculate the tidal tensor at each time step
        txx = []
        tzz = []
        ttensor = TidalTensor(pot)
        for x, y, z in zip(xs, ys, zs):
            tt = ttensor(x, y, z)
            txx.append(tt[0, 0])
            tzz.append(tt[2, 2])

        return np.mean(txx), np.mean(tzz)

    def _setup_inner_interpolation(self):
        """Set up an object used to interpolate the inner orbital components of
        this KeplerRing. The object consists of a list of splines used to
        interpolate each coordinate.

        Returns
        -------
        None
        """
        ex, ey, ez = self._e.T
        jx, jy, jz = self._j.T
        self._interpolatedInner = _setup_splines(self._t, ex=ex, ey=ey, ez=ez,
                                                 jx=jx, jy=jy, jz=jz)

    def _setup_outer_interpolation(self):
        """Set up an object used to interpolate the position and velocity of the
        barycentre of this KeplerRing. The object consists of a list of splines
        used to interpolate each coordinate.

        Returns
        -------
        None
        """
        R, z, phi = self._r.T
        v_R, v_z, v_phi = self._v.T

        # Interpolate x and y rather than phi to avoid phase wrapping issues
        x = R * np.cos(phi)
        y = R * np.sin(phi)

        self._interpolatedOuter = _setup_splines(self._t, x=x, y=y, z=z,
                                                 v_R=v_R, v_z=v_z, v_phi=v_phi)


class KeplerRingError(Exception):
    pass


class KeplerRingWarning(Warning):
    pass


def _setup_splines(t, **kwargs):
    """Return a dictionary of splines used to interpolate a set of parameters.

    Parameters
    ----------
    t : array_like
        1-D array of values of the independent variable.
    kwargs : dict of array_like
        1-D arrays of dependent variables with the same size as t.

    Returns
    -------
    Dictionary containing splines for each provided kwarg. The keys are the
    keyword names, and the values are the splines.
    """
    return {k: InterpolatedUnivariateSpline(t, v) for k, v in kwargs.items()}
