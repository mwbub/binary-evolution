import numpy as np


def elements_to_vectors(ecc, inc, long_asc, arg_peri):
    """Convert from classical orbital elements to vectors.

    Parameters
    ----------
    ecc : array_like
        Eccentricity. Must be between 0 and 1.
    inc : array_like
        Inclination relative to the x-y plane in radians.
    long_asc : array_like
        Longitude of the ascending node in radians.
    arg_peri : array_like
        Argument of pericentre in radians.

    Returns
    -------
    e : ndarray
        The eccentricity vector. If the input parameters are scalars, this has
        shape (3,). Otherwise has shape (S, 3) where S is the shape of the
        input.
    j : ndarray
        The dimensionless angular momentum vector. If this input parameters are
        scalars, this has shape (3,). Otherwise has shape (S, 3) where S is the
        shape of the input.
    """
    ecc = np.array(ecc)
    inc = np.array(inc)
    arg_peri = np.array(arg_peri)
    long_asc = np.array(long_asc)

    if not np.all((0 < ecc) & (ecc < 1)):
        raise ValueError("Eccentricity must be between 0 and 1")

    sin_i = np.sin(inc)
    cos_i = np.cos(inc)
    sin_omega = np.sin(arg_peri)
    cos_omega = np.cos(arg_peri)
    sin_Omega = np.sin(long_asc)
    cos_Omega = np.cos(long_asc)

    j_mag = (1 - ecc**2)**0.5
    jx = j_mag * (sin_i * sin_Omega)
    jy = j_mag * (-sin_i * cos_Omega)
    jz = j_mag * cos_i

    ex = ecc * (cos_omega * cos_Omega - sin_omega * sin_Omega * cos_i)
    ey = ecc * (cos_omega * sin_Omega + sin_omega * cos_Omega * cos_i)
    ez = ecc * (sin_i * sin_omega)

    j = np.stack((jx, jy, jz), axis=-1)
    e = np.stack((ex, ey, ez), axis=-1)

    return e, j


def vectors_to_elements(e, j):
    """Convert from vectors to classical orbital elements.

    Parameters
    ----------
    e : array_like
        Array of eccentricity vectors. The final axis must have size 3, and each
        vector must be orthogonal to and mutually normal with its associated j
        vector (see notes).
    j : array_like
        Array of dimensionless angular momentum vectors. The final axis must
        have size 3, and each vector must be orthogonal to and mutually normal
        with its associated e vector (see notes).

    Returns
    -------
    ecc : array_like
        The eccentricity. Scalar if e has shape (3,), otherwise has the same
        shape as the e parameter, minus the final axis.
    inc : array_like
        The inclination relative to the x-y plane in radians. Scalar if j has
        shape (3,), otherwise has the same shape as the j parameter, minus the
        final axis.
    long_asc : array_like
        The longitude of the ascending node in radians. Scalar if j has shape
        (3,), otherwise has the same shape as the j parameter, minus the final
        axis.
    arg_peri : array_like
        The argument of pericentre in radians. Scalar if e and j have shape
        (3,), otherwise has the same shape as these parameters, minus the final
        axis.

    Notes
    -----
    It is expected that the e and j vectors are orthogonal (e dot j = 0) and
    mutually normal (|e|^2 + |j|^2 = 1). This condition is not enforced,
    however, and may result in exceptions or unexpected behaviour.
    """
    e_shape = np.shape(e)
    j_shape = np.shape(j)

    if len(e_shape) == 0 or len(j_shape) == 0:
        raise ValueError("e and j cannot be scalars")

    if e_shape[-1] != 3 or j_shape[-1] != 3:
        raise ValueError("The final axis of e and j must have size 3")

    e = np.array(e)
    j = np.array(j)

    ex, ey, ez = np.moveaxis(e, -1, 0)
    jx, jy, jz = np.moveaxis(j, -1, 0)

    ecc = (ex**2 + ey**2 + ez**2)**0.5
    inc = np.arctan2((jx**2 + jy**2)**0.5, jz)
    long_asc = np.arctan2(jx, -jy)

    sin_i = np.sin(inc)
    cos_i = np.cos(inc)
    sin_Omega = np.sin(long_asc)
    cos_Omega = np.cos(long_asc)

    arg_peri = np.arctan2(-ex * sin_Omega * cos_i + ey * cos_Omega * cos_i
                          + ez * sin_i, ex * cos_Omega + ey * sin_Omega)

    return ecc, inc, long_asc, arg_peri
