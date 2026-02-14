"""
Functions related to the phase congruency model of feature perception
and phase based approaches to image processing.

Copyright (c) 2015-2018 Peter Kovesi
peterkovesi.com

MIT License.
"""

import numpy as np
from numpy.fft import fft2, ifft2

from .frequencyfilt import (
    filtergrids,
    filtergrid,
    gridangles,
    loggabor,
    lowpassfilter,
    monogenicfilters,
    packedmonogenicfilters,
    cosineangularfilter,
    gaussianangularfilter,
)
from .utilities import histtruncate


def _build_histogram(data, nbins=256):
    """Build histogram of data values.

    Parameters
    ----------
    data : ndarray
        Data values.
    nbins : int
        Number of bins.

    Returns
    -------
    edges : ndarray
        Bin edges.
    counts : ndarray
        Bin counts.
    """
    counts, edges = np.histogram(data.ravel(), bins=nbins)
    return edges, counts


def _rayleighmode(X, nbins=50):
    """Compute mode of data assumed to come from a Rayleigh distribution.

    Parameters
    ----------
    X : ndarray
        Data assumed to come from a Rayleigh distribution.
    nbins : int
        Number of bins for histogram. Default is 50.

    Returns
    -------
    rmode : float
        Estimated mode of the distribution.
    """
    edges, counts = _build_histogram(X, nbins=nbins)
    ind = np.argmax(counts)
    return (edges[ind] + edges[ind + 1]) / 2


def ppdrc(img, wavelength, clip=0.01, n=2):
    """Phase Preserving Dynamic Range Compression.

    Generates a series of dynamic range compressed images at different scales.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be processed.
    wavelength : float or array-like
        Scalar value or array of wavelengths in pixels of the cut-in
        frequencies for the highpass versions of the image.
    clip : float, optional
        Percentage of output image histogram to clip. Default is 0.01%.
    n : int, optional
        Order of the Butterworth high pass filter. Default is 2.

    Returns
    -------
    dimg : ndarray or list of ndarray
        The dynamic range reduced image(s). If only one wavelength is
        specified, the image is returned directly.

    Notes
    -----
    Scaling of the image affects the results. If your image has values of
    order 1 or less it is useful to scale the image up.
    """
    img = np.asarray(img, dtype=np.float64)
    wavelength = np.atleast_1d(np.asarray(wavelength, dtype=np.float64))
    nscale = len(wavelength)

    ph, _, E = highpassmonogenic(img, wavelength, n)

    dimg = [None] * nscale

    if nscale == 1:
        dimg[0] = histtruncate(
            np.sin(ph) * np.log1p(E), clip, clip
        )
    else:
        ranges = np.zeros(nscale)
        for k in range(nscale):
            dimg[k] = histtruncate(
                np.sin(ph[k]) * np.log1p(E[k]), clip, clip
            )
            ranges[k] = np.max(np.abs(dimg[k]))

        maxrange = np.max(ranges)
        for k in range(nscale):
            dimg[k].ravel()[0] = maxrange
            dimg[k].ravel()[1] = -maxrange

    if nscale == 1:
        return dimg[0]
    else:
        return dimg


def highpassmonogenic(img, maxwavelength, n):
    """Compute phase and amplitude in highpass images via monogenic filters.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be processed.
    maxwavelength : float or array-like
        Wavelength(s) in pixels of the cut-in frequency(ies) of the
        Butterworth highpass filter.
    n : int
        The order of the Butterworth filter (>= 1).

    Returns
    -------
    phase : ndarray or list of ndarray
        The local phase. Values between -pi/2 and pi/2.
    orient : ndarray or list of ndarray
        The local orientation. Values between -pi and pi.
    E : ndarray or list of ndarray
        Local energy/amplitude of the signal.

    Notes
    -----
    If maxwavelength is an array, the outputs will be lists of arrays.
    """
    img = np.asarray(img, dtype=np.float64)
    maxwavelength = np.atleast_1d(np.asarray(maxwavelength, dtype=np.float64))

    if np.min(maxwavelength) < 2:
        raise ValueError("Minimum wavelength that can be specified is 2 pixels")

    nscales = len(maxwavelength)
    IMG = fft2(img)

    H1, H2, freq = monogenicfilters(img.shape)

    phase = [None] * nscales
    orient = [None] * nscales
    E = [None] * nscales

    for s in range(nscales):
        # High pass Butterworth filter
        H = 1.0 - 1.0 / (1.0 + (freq * maxwavelength[s]) ** (2 * n))

        f = np.real(ifft2(H * IMG))
        h1f = np.real(ifft2(H * H1 * IMG))
        h2f = np.real(ifft2(H * H2 * IMG))

        phase[s] = np.arctan2(f, np.sqrt(h1f**2 + h2f**2 + np.finfo(float).eps))
        orient[s] = np.arctan2(h2f, h1f)
        E[s] = np.sqrt(f**2 + h1f**2 + h2f**2)

    if nscales == 1:
        return phase[0], orient[0], E[0]
    else:
        return phase, orient, E


def bandpassmonogenic(img, minwavelength, maxwavelength, n):
    """Compute phase and amplitude in bandpass images via monogenic filters.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be processed.
    minwavelength : float or array-like
        Wavelength(s) in pixels of the lower cut frequency.
    maxwavelength : float or array-like
        Wavelength(s) in pixels of the upper cut frequency.
    n : int
        The order of the Butterworth filter (>= 1).

    Returns
    -------
    phase : ndarray or list of ndarray
        The local phase. Values between -pi/2 and pi/2.
    orient : ndarray or list of ndarray
        The local orientation. Values between -pi and pi.
    E : ndarray or list of ndarray
        Local energy/amplitude of the signal.
    """
    img = np.asarray(img, dtype=np.float64)
    minwavelength = np.atleast_1d(np.asarray(minwavelength, dtype=np.float64))
    maxwavelength = np.atleast_1d(np.asarray(maxwavelength, dtype=np.float64))

    if np.min(minwavelength) < 2 or np.min(maxwavelength) < 2:
        raise ValueError("Minimum wavelength that can be specified is 2 pixels")

    if len(minwavelength) != len(maxwavelength):
        raise ValueError("Arrays of min and max wavelengths must be of same length")

    nscales = len(maxwavelength)
    IMG = fft2(img)

    H1, H2, freq = monogenicfilters(img.shape)

    phase = [None] * nscales
    orient = [None] * nscales
    E = [None] * nscales

    for s in range(nscales):
        # Bandpass Butterworth filter
        H = (
            1.0 / (1.0 + (freq * minwavelength[s]) ** (2 * n))
            - 1.0 / (1.0 + (freq * maxwavelength[s]) ** (2 * n))
        )

        f = np.real(ifft2(H * IMG))
        h1f = np.real(ifft2(H * H1 * IMG))
        h2f = np.real(ifft2(H * H2 * IMG))

        phase[s] = np.arctan2(f, np.sqrt(h1f**2 + h2f**2 + np.finfo(float).eps))
        orient[s] = np.arctan2(h2f, h1f)
        E[s] = np.sqrt(f**2 + h1f**2 + h2f**2)

    if nscales == 1:
        return phase[0], orient[0], E[0]
    else:
        return phase, orient, E


def phasecongmono(img, nscale=4, minwavelength=3, mult=2.1, sigmaonf=0.55,
                  k=3.0, noisemethod=-1, cutoff=0.5, g=10.0,
                  deviationgain=1.5):
    """Phase congruency of an image using monogenic filters.

    This code is considerably faster than phasecong3() but you may prefer
    the output from phasecong3()'s oriented filters.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be processed.
    nscale : int
        Number of wavelet scales. Default is 4.
    minwavelength : float
        Wavelength of smallest scale filter. Default is 3.
    mult : float
        Scaling factor between successive filters. Default is 2.1.
    sigmaonf : float
        Ratio of the standard deviation of the Gaussian describing the log
        Gabor filter's transfer function to the filter center frequency.
        Default is 0.55.
    k : float
        No of standard deviations of the noise energy beyond the mean at
        which we set the noise threshold. Default is 3.0.
    noisemethod : float
        Method used to determine noise statistics.
        -1: use median of smallest scale filter responses
        -2: use mode of smallest scale filter responses
        0+: use noisemethod value as the fixed noise threshold.
        Default is -1.
    cutoff : float
        Fractional measure of frequency spread below which phase congruency
        values get penalized. Default is 0.5.
    g : float
        Controls sharpness of the sigmoid function used to weight phase
        congruency for frequency spread. Default is 10.
    deviationgain : float
        Amplification to apply to the calculated phase deviation result.
        Default is 1.5.

    Returns
    -------
    PC : ndarray
        Phase congruency indicating edge significance.
    or_ : ndarray
        Orientation image in radians (-pi/2 to pi/2).
    ft : ndarray
        Local weighted mean phase angle.
    T : float
        Calculated noise threshold.
    """
    img = np.asarray(img, dtype=np.float64)
    epsilon = 0.0001

    rows, cols = img.shape
    IMG = fft2(img)

    sumAn = np.zeros((rows, cols))
    sumf = np.zeros((rows, cols))
    sumh1 = np.zeros((rows, cols))
    sumh2 = np.zeros((rows, cols))
    maxAn = np.zeros((rows, cols))

    tau = 0.0
    T = 0.0

    H, freq = packedmonogenicfilters(rows, cols)

    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        # Construct bandpassed image in frequency domain
        IMGF = np.zeros((rows, cols), dtype=complex)
        for idx in np.ndindex(rows, cols):
            IMGF[idx] = (
                IMG[idx]
                * loggabor(freq[idx], fo, sigmaonf)
                * lowpassfilter(freq[idx], 0.45, 15)
            )

        f = np.real(ifft2(IMGF))
        h = ifft2(IMGF * H)

        h1 = np.real(h)
        h2 = np.imag(h)
        An = np.sqrt(f**2 + h1**2 + h2**2)
        sumAn += An
        sumf += f
        sumh1 += h1
        sumh2 += h2

        if s == 0:
            if abs(noisemethod + 1) < epsilon:
                tau = np.median(sumAn) / np.sqrt(np.log(4))
            elif abs(noisemethod + 2) < epsilon:
                tau = _rayleighmode(sumAn)
            maxAn = An.copy()
        else:
            maxAn = np.maximum(maxAn, An)

    # Frequency spread weighting
    width = (sumAn / (maxAn + epsilon) - 1) / (nscale - 1)
    weight = 1.0 / (1 + np.exp((cutoff - width) * g))

    # Noise threshold
    if noisemethod >= 0:
        T = noisemethod
    else:
        totalTau = tau * (1 - (1 / mult)**nscale) / (1 - (1 / mult))
        EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
        EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
        T = EstNoiseEnergyMean + k * EstNoiseEnergySigma

    # Final computation
    # Julia uses single-arg atan(ratio) giving range (-pi/2, pi/2)
    or_ = np.arctan(-sumh2 / sumh1)
    ft = np.arctan2(sumf, np.sqrt(sumh1**2 + sumh2**2))
    energy = np.sqrt(sumf**2 + sumh1**2 + sumh2**2)

    PC = (
        weight
        * np.maximum(1 - deviationgain * np.arccos(energy / (sumAn + epsilon)), 0)
        * np.maximum(energy - T, 0)
        / (energy + epsilon)
    )

    return PC, or_, ft, T


def phasesymmono(img, nscale=5, minwavelength=3, mult=2.1, sigmaonf=0.55,
                 k=2.0, polarity=0, noisemethod=-1):
    """Phase symmetry of an image using monogenic filters.

    This function calculates the phase symmetry of points in an image.
    This is a contrast invariant measure of symmetry. Can be used as a line
    and blob detector.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be processed.
    nscale : int
        Number of wavelet scales. Default is 5.
    minwavelength : float
        Wavelength of smallest scale filter. Default is 3.
    mult : float
        Scaling factor between successive filters. Default is 2.1.
    sigmaonf : float
        Ratio of the standard deviation of the Gaussian describing the log
        Gabor filter's transfer function. Default is 0.55.
    k : float
        No of standard deviations of the noise energy beyond the mean.
        Default is 2.0.
    polarity : int
        Controls polarity of features to find. 1: bright, -1: dark,
        0: both. Default is 0.
    noisemethod : float
        Noise estimation method. Default is -1.

    Returns
    -------
    phSym : ndarray
        Phase symmetry image (values between 0 and 1).
    symmetryEnergy : ndarray
        Un-normalised raw symmetry energy.
    T : float
        Calculated noise threshold.
    """
    img = np.asarray(img, dtype=np.float64)
    epsilon = 0.0001

    rows, cols = img.shape
    IMG = fft2(img)

    tau = 0.0
    symmetryEnergy = np.zeros((rows, cols))
    sumAn = np.zeros((rows, cols))

    H, freq = packedmonogenicfilters(rows, cols)

    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        IMGF = np.zeros((rows, cols), dtype=complex)
        for idx in np.ndindex(rows, cols):
            IMGF[idx] = (
                IMG[idx]
                * loggabor(freq[idx], fo, sigmaonf)
                * lowpassfilter(freq[idx], 0.4, 10)
            )

        f = np.real(ifft2(IMGF))
        h = ifft2(IMGF * H)

        h1 = np.real(h)
        h2 = np.imag(h)
        hAmp2 = h1**2 + h2**2
        sumAn += np.sqrt(f**2 + hAmp2)

        if polarity == 0:
            symmetryEnergy += np.abs(f) - np.sqrt(hAmp2)
        elif polarity == 1:
            symmetryEnergy += f - np.sqrt(hAmp2)
        elif polarity == -1:
            symmetryEnergy += -f - np.sqrt(hAmp2)

        if s == 0:
            if abs(noisemethod + 1) < epsilon:
                tau = np.median(sumAn) / np.sqrt(np.log(4))
            elif abs(noisemethod + 2) < epsilon:
                tau = _rayleighmode(sumAn)

    # Noise compensation
    if noisemethod >= 0:
        T = noisemethod
    else:
        totalTau = tau * (1 - (1 / mult)**nscale) / (1 - (1 / mult))
        EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
        EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
        T = max(EstNoiseEnergyMean + k * EstNoiseEnergySigma, epsilon)

    phSym = np.maximum(symmetryEnergy - T, 0) / (sumAn + epsilon)

    return phSym, symmetryEnergy, T


def monofilt(img, nscale, minWaveLength, mult, sigmaOnf, orientWrap=False):
    """Apply monogenic filters to an image to obtain 2D analytic signal.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be convolved.
    nscale : int
        Number of filter scales.
    minWaveLength : float
        Wavelength of smallest scale filter.
    mult : float
        Scaling factor between successive filters.
    sigmaOnf : float
        Ratio of the standard deviation of the Gaussian describing the log
        Gabor filter's transfer function.
    orientWrap : bool, optional
        If True, wrap orientation from -pi..pi to 0..pi. Default is False.

    Returns
    -------
    f : list of ndarray
        Bandpass filter responses for each scale.
    h1f : list of ndarray
        Bandpass h1 filter responses for each scale.
    h2f : list of ndarray
        Bandpass h2 filter responses for each scale.
    A : list of ndarray
        Monogenic energy responses for each scale.
    theta : list of ndarray
        Phase orientation responses for each scale.
    psi : list of ndarray
        Phase angle responses for each scale.
    """
    img = np.asarray(img, dtype=np.float64)
    rows, cols = img.shape
    IMG = fft2(img)

    H1, H2, freq = monogenicfilters(rows, cols)

    f_out = [None] * nscale
    h1f_out = [None] * nscale
    h2f_out = [None] * nscale
    A = [None] * nscale
    theta = [None] * nscale
    psi = [None] * nscale

    for s in range(nscale):
        wavelength = minWaveLength * mult**s
        fo = 1.0 / wavelength

        # Vectorized log Gabor filter
        logGabor = np.zeros((rows, cols))
        for idx in np.ndindex(rows, cols):
            logGabor[idx] = loggabor(freq[idx], fo, sigmaOnf)

        H1s = H1 * logGabor
        H2s = H2 * logGabor

        f_out[s] = np.real(ifft2(IMG * logGabor))
        h1f_out[s] = np.real(ifft2(IMG * H1s))
        h2f_out[s] = np.real(ifft2(IMG * H2s))

        A[s] = np.sqrt(f_out[s]**2 + h1f_out[s]**2 + h2f_out[s]**2)
        theta[s] = np.arctan2(h2f_out[s], h1f_out[s])
        psi[s] = np.arctan2(f_out[s], np.sqrt(h1f_out[s]**2 + h2f_out[s]**2))

        if orientWrap:
            theta[s][theta[s] < 0] += np.pi

    return f_out, h1f_out, h2f_out, A, theta, psi


def gaborconvolve(img, nscale, norient, minWaveLength, mult, sigmaOnf,
                  dThetaOnSigma, Lnorm=0):
    """Convolve an image with a bank of log-Gabor filters.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be convolved.
    nscale : int
        Number of wavelet scales.
    norient : int
        Number of filter orientations.
    minWaveLength : float
        Wavelength of smallest scale filter.
    mult : float
        Scaling factor between successive filters.
    sigmaOnf : float
        Ratio of the standard deviation of the Gaussian describing the log
        Gabor filter's transfer function.
    dThetaOnSigma : float
        Ratio of angular interval between filter orientations and the
        standard deviation of the angular Gaussian function.
    Lnorm : int, optional
        Normalization type (0: none, 1: L1, 2: L2). Default is 0.

    Returns
    -------
    EO : list of list of ndarray (complex)
        2D array of complex valued convolution results. EO[s][o] is the
        result for scale s and orientation o.
    BP : list of ndarray
        Bandpass images corresponding to each scale.
    """
    img = np.asarray(img, dtype=np.float64)
    if Lnorm not in [0, 1, 2]:
        raise ValueError("Lnorm must be 0, 1, or 2")

    rows, cols = img.shape
    IMG = fft2(img)
    EO = [[None] * norient for _ in range(nscale)]
    BP = [None] * nscale
    logGabor_arr = [None] * nscale

    freq, fx, fy = filtergrids(rows, cols)
    sintheta, costheta = gridangles(freq, fx, fy)

    thetaSigma = np.pi / norient / dThetaOnSigma

    # Construct radial filter components
    for s in range(nscale):
        wavelength = minWaveLength * mult**s
        fo = 1.0 / wavelength

        lg = np.zeros((rows, cols))
        for idx in np.ndindex(rows, cols):
            lg[idx] = (
                loggabor(freq[idx], fo, sigmaOnf)
                * lowpassfilter(freq[idx], 0.45, 15)
            )

        if Lnorm == 2:
            L = np.sqrt(np.sum(lg**2))
        elif Lnorm == 1:
            L = np.sum(np.abs(np.real(ifft2(lg))))
        else:
            L = 1

        lg /= L
        logGabor_arr[s] = lg
        BP[s] = np.real(ifft2(IMG * lg))

    # Main loop
    for o in range(norient):
        angl = o * np.pi / norient
        angfilter = gaussianangularfilter(angl, thetaSigma, sintheta, costheta)

        for s in range(nscale):
            filt = logGabor_arr[s] * angfilter

            if Lnorm == 2:
                L = np.sqrt(np.sum(np.real(filt)**2 + np.imag(filt)**2)) / np.sqrt(2)
            elif Lnorm == 1:
                L = np.sum(np.abs(np.real(ifft2(filt))))
            else:
                L = 1

            filt = filt / L
            EO[s][o] = ifft2(IMG * filt)

    return EO, BP


def phasecong3(img, nscale=4, norient=6, minwavelength=3, mult=2.1,
               sigmaonf=0.55, k=2.0, cutoff=0.5, g=10.0, noisemethod=-1):
    """Compute edge and corner phase congruency via log-Gabor filters.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be processed.
    nscale : int
        Number of wavelet scales. Default is 4.
    norient : int
        Number of filter orientations. Default is 6.
    minwavelength : float
        Wavelength of smallest scale filter. Default is 3.
    mult : float
        Scaling factor between successive filters. Default is 2.1.
    sigmaonf : float
        Ratio of the standard deviation of the log Gabor filter's transfer
        function. Default is 0.55.
    k : float
        No of standard deviations of noise energy beyond the mean. Default 2.
    cutoff : float
        Fractional measure of frequency spread. Default is 0.5.
    g : float
        Sharpness of the sigmoid function. Default is 10.
    noisemethod : float
        Noise estimation method. Default is -1.

    Returns
    -------
    M : ndarray
        Maximum moment of phase congruency covariance (edge strength).
    m : ndarray
        Minimum moment of phase congruency covariance (corner strength).
    or_ : ndarray
        Orientation image in radians (-pi/2 to pi/2).
    featType : ndarray
        Local weighted mean phase angle (feature type).
    EO : list of list of ndarray (complex)
        2D array of complex valued convolution results.
    T : float
        Calculated noise threshold.
    """
    img = np.asarray(img, dtype=np.float64)
    epsilon = 1e-5
    rows, cols = img.shape
    IMG = fft2(img)

    logGabor_arr = [None] * nscale
    EO = [[None] * norient for _ in range(nscale)]
    EnergyV = np.zeros((rows, cols, 3))

    covx2 = np.zeros((rows, cols))
    covy2 = np.zeros((rows, cols))
    covxy = np.zeros((rows, cols))

    sumE_ThisOrient = np.zeros((rows, cols))
    sumO_ThisOrient = np.zeros((rows, cols))
    sumAn_ThisOrient = np.zeros((rows, cols))
    Energy = np.zeros((rows, cols))
    MeanE = np.zeros((rows, cols))
    MeanO = np.zeros((rows, cols))
    An = np.zeros((rows, cols))
    maxAn = np.zeros((rows, cols))

    M = np.zeros((rows, cols))
    m = np.zeros((rows, cols))

    T = 0.0
    tau = 0.0

    freq, fx, fy = filtergrids(rows, cols)
    sintheta, costheta = gridangles(freq, fx, fy)

    # Construct radial filter components
    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        lg = np.zeros((rows, cols))
        for idx in np.ndindex(rows, cols):
            lg[idx] = (
                loggabor(freq[idx], fo, sigmaonf)
                * lowpassfilter(freq[idx], 0.45, 15)
            )
        logGabor_arr[s] = lg

    # Main loop
    for o in range(norient):
        angl = o * np.pi / norient
        wavelen = 4 * np.pi / norient
        angfilter = cosineangularfilter(angl, wavelen, sintheta, costheta)

        sumE_ThisOrient[:] = 0
        sumO_ThisOrient[:] = 0
        sumAn_ThisOrient[:] = 0
        Energy[:] = 0

        for s in range(nscale):
            filt = logGabor_arr[s] * angfilter
            EO[s][o] = ifft2(IMG * filt)

            An[:] = np.abs(EO[s][o])
            sumAn_ThisOrient += An
            sumE_ThisOrient += np.real(EO[s][o])
            sumO_ThisOrient += np.imag(EO[s][o])

            if s == 0:
                if abs(noisemethod + 1) < epsilon:
                    tau = np.median(sumAn_ThisOrient) / np.sqrt(np.log(4))
                elif abs(noisemethod + 2) < epsilon:
                    tau = _rayleighmode(sumAn_ThisOrient)
                maxAn[:] = An
            else:
                maxAn = np.maximum(maxAn, An)

        # Accumulate total 3D energy vector
        EnergyV[:, :, 0] += sumE_ThisOrient
        EnergyV[:, :, 1] += np.cos(angl) * sumO_ThisOrient
        EnergyV[:, :, 2] += np.sin(angl) * sumO_ThisOrient

        # Weighted mean filter response vector
        XEnergy = np.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        # Phase congruency energy
        for s in range(nscale):
            E_val = np.real(EO[s][o])
            O_val = np.imag(EO[s][o])
            Energy += (
                E_val * MeanE + O_val * MeanO
                - np.abs(E_val * MeanO - O_val * MeanE)
            )

        # Noise threshold
        if noisemethod >= 0:
            T = noisemethod
        else:
            totalTau = tau * (1 - (1 / mult)**nscale) / (1 - (1 / mult))
            EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
            EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
            T = EstNoiseEnergyMean + k * EstNoiseEnergySigma

        Energy = np.maximum(Energy - T, 0)

        # Frequency spread weighting and phase congruency
        width = (sumAn_ThisOrient / (maxAn + epsilon) - 1) / (nscale - 1)
        weight_arr = 1.0 / (1 + np.exp((cutoff - width) * g))
        PCo = weight_arr * Energy / sumAn_ThisOrient

        covx = PCo * np.cos(angl)
        covy = PCo * np.sin(angl)
        covx2 += covx**2
        covy2 += covy**2
        covxy += covx * covy

    # Edge and corner calculations
    covx2 /= (norient / 2)
    covy2 /= (norient / 2)
    covxy *= 4 / norient

    denom = np.sqrt(covxy**2 + (covx2 - covy2)**2) + epsilon
    M = (covy2 + covx2 + denom) / 2
    m = (covy2 + covx2 - denom) / 2

    # Julia uses single-arg atan(ratio) giving range (-pi/2, pi/2)
    or_ = np.arctan(-EnergyV[:, :, 2] / EnergyV[:, :, 1])
    OddV = np.sqrt(EnergyV[:, :, 1]**2 + EnergyV[:, :, 2]**2)
    featType = np.arctan2(EnergyV[:, :, 0], OddV)

    return M, m, or_, featType, EO, T


def phasesym(img, nscale=5, norient=6, minwavelength=3, mult=2.1,
             sigmaonf=0.55, k=2.0, polarity=0, noisemethod=-1):
    """Compute phase symmetry on an image via log-Gabor filters.

    This function calculates the phase symmetry of points in an image.
    This is a contrast invariant measure of symmetry.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be processed.
    nscale : int
        Number of wavelet scales. Default is 5.
    norient : int
        Number of filter orientations. Default is 6.
    minwavelength : float
        Wavelength of smallest scale filter. Default is 3.
    mult : float
        Scaling factor between successive filters. Default is 2.1.
    sigmaonf : float
        Ratio of the standard deviation of the log Gabor filter. Default 0.55.
    k : float
        No of standard deviations of noise energy beyond the mean. Default 2.
    polarity : int
        Controls polarity. 1: bright, -1: dark, 0: both. Default is 0.
    noisemethod : float
        Noise estimation method. Default is -1.

    Returns
    -------
    phSym : ndarray
        Phase symmetry image (values between 0 and 1).
    orientation : ndarray
        Orientation image in radians (-pi/2 to pi/2).
    totalEnergy : ndarray
        Un-normalised raw symmetry energy.
    T : float
        Calculated noise threshold.
    """
    img = np.asarray(img, dtype=np.float64)
    epsilon = 1e-4
    rows, cols = img.shape
    IMG = fft2(img)

    logGabor_arr = [None] * nscale
    totalEnergy = np.zeros((rows, cols))
    totalSumAn = np.zeros((rows, cols))
    orientation = np.zeros((rows, cols))
    maxEnergy = np.zeros((rows, cols))
    sumAn_ThisOrient = np.zeros((rows, cols))
    Energy_ThisOrient = np.zeros((rows, cols))
    An = np.zeros((rows, cols))

    tau = 0.0
    T = 0.0

    freq, fx, fy = filtergrids(rows, cols)
    sintheta, costheta = gridangles(freq, fx, fy)

    # Construct radial filter components
    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        lg = np.zeros((rows, cols))
        for idx in np.ndindex(rows, cols):
            lg[idx] = (
                loggabor(freq[idx], fo, sigmaonf)
                * lowpassfilter(freq[idx], 0.45, 15)
            )
        logGabor_arr[s] = lg

    # Main loop
    for o in range(norient):
        angl = o * np.pi / norient
        wavelen = 4 * np.pi / norient
        angfilter = cosineangularfilter(angl, wavelen, sintheta, costheta)

        sumAn_ThisOrient[:] = 0
        Energy_ThisOrient[:] = 0

        for s in range(nscale):
            filt = logGabor_arr[s] * angfilter
            EO = ifft2(IMG * filt)
            An[:] = np.abs(EO)
            sumAn_ThisOrient += An

            if s == 0:
                if abs(noisemethod + 1) < epsilon:
                    tau = np.median(sumAn_ThisOrient) / np.sqrt(np.log(4))
                elif abs(noisemethod + 2) < epsilon:
                    tau = _rayleighmode(sumAn_ThisOrient)

            if polarity == 0:
                Energy_ThisOrient += np.abs(np.real(EO)) - np.abs(np.imag(EO))
            elif polarity == 1:
                Energy_ThisOrient += np.real(EO) - np.abs(np.imag(EO))
            elif polarity == -1:
                Energy_ThisOrient += -np.real(EO) - np.abs(np.imag(EO))

        # Noise threshold
        if noisemethod >= 0:
            T = noisemethod
        else:
            totalTau = tau * (1 - (1 / mult)**nscale) / (1 - (1 / mult))
            EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
            EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
            T = max(EstNoiseEnergyMean + k * EstNoiseEnergySigma, epsilon)

        Energy_ThisOrient -= T

        totalSumAn += sumAn_ThisOrient
        totalEnergy += Energy_ThisOrient

        if o == 0:
            maxEnergy[:] = Energy_ThisOrient
        else:
            mask = Energy_ThisOrient > maxEnergy
            orientation[mask] = o
            maxEnergy[mask] = Energy_ThisOrient[mask]

    phSym = np.maximum(totalEnergy, 0) / (totalSumAn + epsilon)
    orientation = orientation * np.pi / norient - np.pi / 2

    return phSym, orientation, totalEnergy, T


def ppdenoise(img, nscale=5, norient=6, mult=2.5, minwavelength=2,
              sigmaonf=0.55, dthetaonsigma=1.0, k=3.0, softness=1.0):
    """Phase preserving wavelet image denoising.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be processed (greyscale).
    nscale : int
        No of filter scales to use (5-7). Default is 5.
    norient : int
        No of orientations to use. Default is 6.
    mult : float
        Multiplying factor between successive scales. Default is 2.5.
    minwavelength : float
        Wavelength of smallest scale filter. Default is 2.
    sigmaonf : float
        Ratio of log Gabor filter's transfer function. Default is 0.55.
    dthetaonsigma : float
        Ratio of angular interval between filter orientations. Default is 1.0.
    k : float
        No of standard deviations of noise to reject. Default is 3.
    softness : float
        Degree of soft thresholding (0: hard, 1: soft). Default is 1.0.

    Returns
    -------
    cleanimage : ndarray
        Denoised image.
    """
    img = np.asarray(img, dtype=np.float64)
    epsilon = 1e-5

    thetaSigma = np.pi / norient / dthetaonsigma
    rows, cols = img.shape
    IMG = fft2(img)

    freq, fx, fy = filtergrids(rows, cols)
    sintheta, costheta = gridangles(freq, fx, fy)

    totalEnergy = np.zeros((rows, cols), dtype=complex)

    RayMean = 0.0
    RayVar = 0.0

    for o in range(norient):
        angl = o * np.pi / norient
        angfilter = gaussianangularfilter(angl, thetaSigma, sintheta, costheta)

        wavelength = minwavelength

        for s in range(nscale):
            fo = 1.0 / wavelength
            filt = np.zeros((rows, cols))
            for idx in np.ndindex(rows, cols):
                filt[idx] = loggabor(freq[idx], fo, sigmaonf) * angfilter[idx]

            EO = ifft2(IMG * filt)
            aEO = np.abs(EO)

            if s == 0:
                RayMean = np.median(aEO) * 0.5 * np.sqrt(-np.pi / np.log(0.5))
                RayVar = (4 - np.pi) * (RayMean**2) / np.pi

            T = (RayMean + k * np.sqrt(RayVar)) / (mult**s)

            mask = aEO > T
            V = softness * T * EO / (aEO + epsilon)
            EO_denoised = EO.copy()
            EO_denoised[mask] -= V[mask]
            totalEnergy[mask] += EO_denoised[mask]

            wavelength *= mult

    return np.real(totalEnergy)
