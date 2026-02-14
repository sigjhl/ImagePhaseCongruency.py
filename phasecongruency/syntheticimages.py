"""
Functions for creating various synthetic test images for evaluating edge
detectors. Most of these images cause considerable grief for gradient based
operators.

Copyright (c) 2015-2017 Peter Kovesi
peterkovesi.com

MIT License.
"""

import numpy as np
from numpy.fft import fft2, ifft2

from .frequencyfilt import filtergrid
from .utilities import imgnormalize


def step2line(sze=512, nscales=50, ampexponent=-1, ncycles=1.5,
              phasecycles=0.25):
    """A phase congruent test image that interpolates from a step to a line.

    Generates a test image where the feature type changes from a step edge to a
    line feature from top to bottom.

    Parameters
    ----------
    sze : int
        Number of rows in test image. Default is 512.
    nscales : int
        No of Fourier components used to construct the signal. Default is 50.
    ampexponent : float
        Decay exponent of amplitude with frequency. Default is -1.
    ncycles : float
        Number of wave cycles across the width of the image. Default is 1.5.
    phasecycles : float
        Number of feature type phase cycles going vertically down the image.
        Default is 0.25.

    Returns
    -------
    img : ndarray (sze, sze)
        The test image.
    """
    x = np.arange(sze) / (sze - 1) * ncycles * 2 * np.pi

    img = np.zeros((sze, sze))
    phaseoffset = 0.0

    for row in range(sze):
        for scale in range(1, nscales * 2, 2):
            img[row, :] += scale ** float(ampexponent) * np.sin(
                scale * x + phaseoffset
            )
        phaseoffset += phasecycles * 2 * np.pi / sze

    return img


def circsine(sze=512, wavelength=40, nscales=50, ampexponent=-1, offset=0,
             p=2, trim=False):
    """Generate a phase congruent circular sine wave grating.

    Useful for testing the isotropy of response of a feature detector.

    Parameters
    ----------
    sze : int
        The size of the square image to be produced. Default is 512.
    wavelength : float
        The wavelength in pixels of the sine wave. Default is 40.
    nscales : int
        No of Fourier components. Default is 50.
    ampexponent : float
        Decay exponent of amplitude with frequency. Default is -1.
    offset : float
        Angle of phase congruency. Default is 0.
    p : int
        Norm to use in calculating radius. Default is 2 (circular). Must be
        even.
    trim : bool
        Whether to trim the circular pattern from corners. Default is False.

    Returns
    -------
    img : ndarray (sze, sze)
        The test image.
    """
    if p % 2 != 0:
        raise ValueError("p should be an even number")

    if sze % 2 == 0:
        l = -sze / 2
        u = sze / 2 - 1
    else:
        l = -(sze - 1) / 2
        u = (sze - 1) / 2

    coords = np.arange(l, u + 1)
    x, y = np.meshgrid(coords, coords)
    r = (np.abs(x) ** p + np.abs(y) ** p) ** (1.0 / p)

    img = np.zeros(r.shape)
    for scale in range(1, 2 * nscales, 2):
        img += scale ** float(ampexponent) * np.sin(
            scale * r * 2 * np.pi / wavelength + offset
        )

    if trim:
        cycles = np.floor(sze / 2 / wavelength)
        mask = (r < cycles * wavelength).astype(float)
        img *= mask + (1 - mask)

    return img


def starsine(sze=512, ncycles=10, nscales=50, ampexponent=-1, offset=0):
    """Generate a phase congruent star shaped sine wave grating.

    Useful for testing the behaviour of feature detectors at line junctions.

    Parameters
    ----------
    sze : int
        The size of the square image to be produced. Default is 512.
    ncycles : float
        Number of sine wave cycles around centre point. Default is 10.
    nscales : int
        No of Fourier components. Default is 50.
    ampexponent : float
        Decay exponent of amplitude with frequency. Default is -1.
    offset : float
        Angle of phase congruency. Default is 0.

    Returns
    -------
    img : ndarray (sze, sze)
        The test image.
    """
    if sze % 2 == 0:
        l = -sze / 2
        u = sze / 2 - 1
    else:
        l = -(sze - 1) / 2
        u = (sze - 1) / 2

    coords = np.arange(l, u + 1)
    # Julia: [atan(y,x) for x = l:u, y = l:u] puts x in rows, y in columns.
    # To match: row index = x, col index = y, so theta[i,j] = atan2(coords[j], coords[i])
    x, y = np.meshgrid(coords, coords)
    theta = np.arctan2(x, y)

    img = np.zeros(theta.shape)
    for scale in range(1, nscales * 2, 2):
        img += scale ** float(ampexponent) * np.sin(
            scale * ncycles * theta + offset
        )

    return img


def noiseonf(sze, p):
    """Create 1/f^p spectrum noise images.

    Parameters
    ----------
    sze : int or tuple
        A tuple (rows, cols) or single integer specifying size of image.
    p : float
        Exponent of spectrum decay = 1/(f^p).

    Returns
    -------
    img : ndarray (rows, cols)
        The noise image with specified spectrum.

    Notes
    -----
    Reference values for p:
        p = 0   - raw Gaussian noise image.
        p = 1   - gives 1/f 'standard' drop-off for 'natural' images.
        p = 1.5 - seems to give most interesting 'cloud patterns'.
        p > 2   - produces 'blobby' images.
    """
    if isinstance(sze, (int, np.integer)):
        rows, cols = sze, sze
    else:
        rows, cols = sze

    img = np.random.randn(rows, cols)
    imgfft = fft2(img)
    mag = np.abs(imgfft)
    phase = imgfft / (mag + np.finfo(float).eps)

    # Construct amplitude spectrum filter
    radius = filtergrid(rows, cols) * max(rows, cols) + 1
    filt = 1.0 / (radius ** p)

    newfft = filt * phase
    img = np.real(ifft2(newfft))

    return img


def nophase(img):
    """Randomize image phase leaving amplitude spectrum unchanged.

    Parameters
    ----------
    img : ndarray (2D, real)
        Input image.

    Returns
    -------
    newimg : ndarray (2D)
        Image with randomized phase.
    """
    IMG = fft2(img)
    mag = np.abs(IMG)

    phaseAng = np.random.rand(*img.shape) * 2 * np.pi
    phase = np.cos(phaseAng) + 1j * np.sin(phaseAng)

    newfft = mag * phase
    return np.real(ifft2(newfft))


def quantizephase(img, N):
    """Quantize phase values in an image.

    Parameters
    ----------
    img : ndarray (2D, real)
        Image to be processed.
    N : int
        Desired number of quantized phase values.

    Returns
    -------
    qimg : ndarray (2D)
        Phase quantized image.

    Notes
    -----
    Phase values can be quantized very heavily with little perceptual loss.
    The value of N can be as low as 4, or even 3!
    """
    IMG = fft2(img)
    amp = np.abs(IMG)
    phase = np.angle(IMG)

    phase = (
        np.floor((phase + np.pi - 0.001) / (2 * np.pi) * N)
        * (2 * np.pi) / N
        - np.pi
        + np.pi / N
    )

    QIMG = amp * (np.cos(phase) + 1j * np.sin(phase))
    return np.real(ifft2(QIMG))


def swapphase(img1, img2):
    """Demonstrate phase-amplitude swapping between images.

    Parameters
    ----------
    img1 : ndarray (2D, real)
        First input image.
    img2 : ndarray (2D, real)
        Second input image (same size as img1).

    Returns
    -------
    newimg1 : ndarray (2D)
        Image from phase of img1 and magnitude of img2.
    newimg2 : ndarray (2D)
        Image from phase of img2 and magnitude of img1.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must be the same size")

    IMG1 = fft2(img1)
    IMG2 = fft2(img2)
    mag1 = np.abs(IMG1)
    mag2 = np.abs(IMG2)
    phase1 = IMG1 / (mag1 + np.finfo(float).eps)
    phase2 = IMG2 / (mag2 + np.finfo(float).eps)

    NEWIMG1 = mag2 * phase1
    NEWIMG2 = mag1 * phase2
    newimg1 = np.real(ifft2(NEWIMG1))
    newimg2 = np.real(ifft2(NEWIMG2))

    newimg1 = imgnormalize(newimg1)
    newimg2 = imgnormalize(newimg2)

    return newimg1, newimg2
