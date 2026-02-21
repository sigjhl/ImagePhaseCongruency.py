"""
Functions for constructing image filters in the frequency domain.

Copyright (c) Peter Kovesi
peterkovesi.com

MIT License.
"""

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift


def filtergrids(rows, cols=None):
    """Generate grids for constructing frequency domain filters.

    Parameters
    ----------
    rows : int or tuple
        Number of rows, or a tuple (rows, cols).
    cols : int, optional
        Number of columns. Not needed if rows is a tuple.

    Returns
    -------
    f : ndarray
        Grid of size (rows, cols) containing frequency values from 0 to 0.5.
        The grid is quadrant shifted so that 0 frequency is at f[0,0].
    fx : ndarray
        Grid containing normalised frequency values ranging from -0.5 to 0.5
        in x direction. Quadrant shifted.
    fy : ndarray
        Grid containing normalised frequency values ranging from -0.5 to 0.5
        in y direction. Quadrant shifted.
    """
    if cols is None:
        if isinstance(rows, (tuple, list)):
            rows, cols = rows
        else:
            raise ValueError("Must provide both rows and cols, or a tuple (rows, cols)")

    if cols % 2 == 1:
        fxrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / cols
    else:
        fxrange = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2 == 1:
        fyrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / rows
    else:
        fyrange = np.arange(-rows / 2, rows / 2) / rows

    fx, fy = np.meshgrid(fxrange, fyrange)

    # Quadrant shift so that filters are constructed with 0 frequency at corners
    fx = ifftshift(fx)
    fy = ifftshift(fy)

    f = np.sqrt(fx**2 + fy**2)
    return f, fx, fy


def filtergrid(rows, cols=None):
    """Generate grid for constructing frequency domain filters.

    Parameters
    ----------
    rows : int or tuple
        Number of rows, or a tuple (rows, cols).
    cols : int, optional
        Number of columns.

    Returns
    -------
    f : ndarray
        Grid of size (rows, cols) containing normalised frequency values
        from 0 to 0.5. Grid is quadrant shifted so that 0 frequency is at
        f[0,0].
    """
    if cols is None:
        if isinstance(rows, (tuple, list)):
            rows, cols = rows
        else:
            raise ValueError("Must provide both rows and cols, or a tuple (rows, cols)")

    if cols % 2 == 1:
        fxrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / cols
    else:
        fxrange = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2 == 1:
        fyrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / rows
    else:
        fyrange = np.arange(-rows / 2, rows / 2) / rows

    fx, fy = np.meshgrid(fxrange, fyrange)
    f = np.sqrt(fx**2 + fy**2)
    return ifftshift(f)


def monogenicfilters(rows, cols=None):
    """Generate monogenic filter grids.

    Parameters
    ----------
    rows : int or tuple
        Number of rows, or a tuple (rows, cols).
    cols : int, optional
        Number of columns.

    Returns
    -------
    H1 : ndarray (complex)
        First monogenic filter: 1j * fx / f
    H2 : ndarray (complex)
        Second monogenic filter: 1j * fy / f
    f : ndarray
        Frequency grid corresponding to the filters.

    Notes
    -----
    H1, H2, and f are quadrant shifted so that the 0 frequency value is at
    coordinate [0,0].
    """
    if cols is None:
        if isinstance(rows, (tuple, list)):
            rows, cols = rows
        else:
            raise ValueError("Must provide both rows and cols, or a tuple (rows, cols)")

    f, fx, fy = filtergrids(rows, cols)
    f[0, 0] = 1  # Avoid divide by zero

    H1 = 1j * fx / f
    H2 = 1j * fy / f

    H1[0, 0] = 0  # Restore 0 DC value
    H2[0, 0] = 0
    f[0, 0] = 0
    return H1, H2, f


def packedmonogenicfilters(rows, cols=None):
    """Monogenic filter where both filters are packed in one Complex grid.

    Parameters
    ----------
    rows : int or tuple
        Number of rows, or a tuple (rows, cols).
    cols : int, optional
        Number of columns.

    Returns
    -------
    H : ndarray (complex)
        The two monogenic filters packed into one complex grid.
    f : ndarray
        Frequency grid corresponding to the filter.

    Notes
    -----
    The two monogenic filters H1 = 1j*fx/f and H2 = 1j*fy/f are packed
    together as a complex valued matrix. When the convolution is performed
    via the FFT the real part of the result will correspond to the convolution
    with H1 and the imaginary part with H2.
    """
    if cols is None:
        if isinstance(rows, (tuple, list)):
            rows, cols = rows
        else:
            raise ValueError("Must provide both rows and cols, or a tuple (rows, cols)")

    f, fx, fy = filtergrids(rows, cols)
    f[0, 0] = 1  # Avoid divide by zero

    # Pack: H = (1j*fx - fy) / f  (note subtraction because i*i = -1)
    H = (1j * fx - fy) / f

    H[0, 0] = 0  # Restore 0 DC value
    f[0, 0] = 0
    return H, f


def lowpassfilter(sze_or_f, cutoff, n):
    """Construct a low-pass Butterworth filter.

    Parameters
    ----------
    sze_or_f : tuple or float
        Either a tuple (rows, cols) specifying the size of filter to construct,
        or a scalar frequency value.
    cutoff : float
        Cutoff frequency of the filter (0-0.5).
    n : int
        Order of the filter (n is doubled so that it is always even).

    Returns
    -------
    f : ndarray or float
        The filter. If sze_or_f was a tuple, returns a 2D array with frequency
        origin at the corners.
    """
    if isinstance(sze_or_f, (tuple, list)):
        if cutoff < 0 or cutoff > 0.5:
            raise ValueError("cutoff frequency must be between 0 and 0.5")
        f = filtergrid(sze_or_f)
        return 1.0 / (1.0 + (f / cutoff) ** (2 * n))
    else:
        return 1.0 / (1.0 + (sze_or_f / cutoff) ** (2 * n))


def highpassfilter(sze, cutoff, n):
    """Construct a high-pass Butterworth filter.

    Parameters
    ----------
    sze : tuple
        A tuple (rows, cols) specifying the size of filter to construct.
    cutoff : float
        Cutoff frequency of the filter (0-0.5).
    n : int
        Order of the filter.

    Returns
    -------
    f : ndarray
        Frequency domain filter with frequency origin at the corners.
    """
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError("cutoff frequency must be between 0 and 0.5")
    return 1.0 - lowpassfilter(sze, cutoff, n)


def bandpassfilter(sze, cutin, cutoff, n):
    """Construct a band-pass Butterworth filter.

    Parameters
    ----------
    sze : tuple
        A tuple (rows, cols) specifying the size of filter to construct.
    cutin : float
        Lower cutoff frequency (0-0.5).
    cutoff : float
        Upper cutoff frequency (0-0.5).
    n : int
        Order of the filter.

    Returns
    -------
    f : ndarray
        Frequency domain filter with frequency origin at the corners.
    """
    if cutin < 0 or cutin > 0.5 or cutoff < 0 or cutoff > 0.5:
        raise ValueError("Frequencies must be between 0 and 0.5")
    if n < 1:
        raise ValueError("Order of filter must be greater than 1")
    return lowpassfilter(sze, cutoff, n) - lowpassfilter(sze, cutin, n)


def highboostfilter(sze, cutoff, n, boost):
    """Construct a high-boost Butterworth filter.

    Parameters
    ----------
    sze : tuple
        A tuple (rows, cols) specifying the size of filter to construct.
    cutoff : float
        Cutoff frequency of the filter (0-0.5).
    n : int
        Order of the filter.
    boost : float
        Ratio that high frequency values are boosted relative to low
        frequency values.

    Returns
    -------
    f : ndarray
        Frequency domain filter with frequency origin at the corners.
    """
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError("cutoff frequency must be between 0 and 0.5")

    if boost >= 1:  # high-boost filter
        f = (1.0 - 1.0 / boost) * highpassfilter(sze, cutoff, n) + 1.0 / boost
    else:  # low-boost filter
        f = (1.0 - boost) * lowpassfilter(sze, cutoff, n) + boost

    return f


def loggabor(f, fo, sigmaOnf):
    """The logarithmic Gabor function in the frequency domain.

    Parameters
    ----------
    f : float or ndarray
        Frequency value(s) to evaluate the function at.
    fo : float
        Centre frequency of filter.
    sigmaOnf : float
        Ratio of the standard deviation of the Gaussian describing the log
        Gabor filter's transfer function in the frequency domain to the filter
        center frequency.

    Returns
    -------
    float or ndarray
        Log Gabor filter value(s), matching the shape of ``f``.

    Notes
    -----
    sigmaOnf = 0.75 gives a filter bandwidth of about 1 octave.
    sigmaOnf = 0.55 gives a filter bandwidth of about 2 octaves.
    """
    f_arr = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f_arr, dtype=np.float64)

    mask = f_arr >= np.finfo(float).eps
    if np.any(mask):
        log_sigma_sq = np.log(sigmaOnf) ** 2
        out[mask] = np.exp((-(np.log(f_arr[mask] / fo)) ** 2) / (2 * log_sigma_sq))

    if np.isscalar(f):
        return float(out)
    return out


def gridangles(freq, fx, fy):
    """Generate arrays of filter grid angles.

    Parameters
    ----------
    freq : ndarray
        Frequency grid (output of filtergrids).
    fx : ndarray
        x-frequency grid (output of filtergrids).
    fy : ndarray
        y-frequency grid (output of filtergrids).

    Returns
    -------
    sintheta : ndarray
        Sine of the angles in the filter grid.
    costheta : ndarray
        Cosine of the angles in the filter grid.
    """
    freq[0, 0] = 1  # Avoid divide by 0
    sintheta = fx / freq
    costheta = fy / freq
    freq[0, 0] = 0  # Restore 0 DC

    return sintheta, costheta


def cosineangularfilter(angl, wavelen, sintheta, costheta):
    """Orientation selective frequency domain filter with cosine windowing.

    Parameters
    ----------
    angl : float
        Orientation of the filter (radians).
    wavelen : float
        Wavelength of the angular cosine window function.
    sintheta : ndarray
        Grid as returned by gridangles().
    costheta : ndarray
        Grid as returned by gridangles().

    Returns
    -------
    fltr : ndarray
        The angular filter.
    """
    sinangl = np.sin(angl)
    cosangl = np.cos(angl)

    ds = sintheta * cosangl - costheta * sinangl
    dc = costheta * cosangl + sintheta * sinangl
    dtheta = np.abs(np.arctan2(ds, dc))
    dtheta = np.minimum(dtheta * 2 * np.pi / wavelen, np.pi)

    return (np.cos(dtheta) + 1) / 2


def gaussianangularfilter(angl, thetaSigma, sintheta, costheta):
    """Orientation selective frequency domain filter with Gaussian windowing.

    Parameters
    ----------
    angl : float
        Orientation of the filter (radians).
    thetaSigma : float
        Standard deviation of angular Gaussian window function.
    sintheta : ndarray
        Grid as returned by gridangles().
    costheta : ndarray
        Grid as returned by gridangles().

    Returns
    -------
    fltr : ndarray
        The angular filter.
    """
    sinangl = np.sin(angl)
    cosangl = np.cos(angl)

    ds = sintheta * cosangl - costheta * sinangl
    dc = costheta * cosangl + sintheta * sinangl
    dtheta = np.arctan2(ds, dc)

    return np.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))


def perfft2(img):
    """2D Fourier transform of Moisan's periodic image component.

    Parameters
    ----------
    img : ndarray (2D, real)
        Image to be transformed.

    Returns
    -------
    P : ndarray (complex)
        2D FFT of periodic image component.
    S : ndarray (complex)
        2D FFT of smooth component.
    p : ndarray (real)
        Periodic component (spatial domain).
    s : ndarray (real)
        Smooth component (spatial domain).

    Notes
    -----
    Moisan's "Periodic plus Smooth Image Decomposition" decomposes an image
    into two components: img = p + s, where s is the 'smooth' component with
    mean 0 and p is the 'periodic' component which has no sharp discontinuities
    when one moves cyclically across the image boundaries.

    Reference:
    L. Moisan, "Periodic plus Smooth Image Decomposition", Journal of
    Mathematical Imaging and Vision, vol 39:2, pp. 161-179, 2011.
    """
    rows, cols = img.shape

    # Compute the boundary image
    s = np.zeros((rows, cols))
    s[0, :] = img[0, :] - img[-1, :]
    s[-1, :] = -s[0, :]
    s[:, 0] = s[:, 0] + img[:, 0] - img[:, -1]
    s[:, -1] = s[:, -1] - img[:, 0] + img[:, -1]

    # Generate grid for the filter in frequency domain
    cxrange = 2 * np.pi * np.arange(cols) / cols
    cyrange = 2 * np.pi * np.arange(rows) / rows
    cx, cy = np.meshgrid(cxrange, cyrange)
    denom = 2 * (2 - np.cos(cx) - np.cos(cy))

    # Generate FFT of smooth component
    S = fft2(s) / denom
    S[0, 0] = 0.0  # Enforce 0 mean

    P = fft2(img) - S  # FFT of periodic component

    s = np.real(ifft2(S))
    p = img - s

    return P, S, p, s


def geoseries(s1, mult_or_n=None, n=None):
    """Generate geometric series.

    Useful for generating geometrically scaled wavelengths for specifying
    filter banks.

    Can be called as:
        geoseries(s1, mult, n) - Generate n values starting at s1 with
                                 multiplier mult.
        geoseries((s1, sn), n) - Generate n values from s1 to sn.

    Parameters
    ----------
    s1 : float or tuple
        Starting value, or tuple (s1, sn) of start and end values.
    mult_or_n : float or int
        If s1 is a scalar, this is the multiplier between successive values.
        If s1 is a tuple, this is n (number of elements).
    n : int, optional
        Number of elements in the series. Only used when s1 is a scalar.

    Returns
    -------
    s : ndarray
        The geometric series.
    """
    if isinstance(s1, (tuple, list)):
        s1_val, sn_val = s1
        n_val = int(mult_or_n)
        assert s1_val > 0, "Starting value must be > 0"
        assert n_val > 0, "Number of elements must be a +ve integer"
        if n_val == 1:
            return np.array([s1_val], dtype=float)
        mult = np.exp(np.log(sn_val / s1_val) / (n_val - 1))
        return s1_val * mult ** np.arange(n_val)
    else:
        assert n is not None, "Must provide n when s1 is a scalar"
        assert n > 0, "Number of elements must be a +ve integer"
        assert s1 > 0, "Starting value must be > 0"
        return s1 * mult_or_n ** np.arange(n)
