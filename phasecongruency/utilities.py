"""
Utility functions for ImagePhaseCongruency.

Copyright (c) 2015 Peter Kovesi
peterkovesi.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

The Software is provided "as is", without warranty of any kind.
"""

import warnings
import numpy as np
from scipy.ndimage import label, distance_transform_edt


def fillnan(img):
    """Fill NaN values in an image with closest non-NaN value.

    This can be used as a crude (but quick) 'inpainting' function to allow
    a FFT to be computed on an image containing NaN values.

    Parameters
    ----------
    img : ndarray
        Image to be filled.

    Returns
    -------
    newimg : ndarray
        Filled image.
    mask : ndarray of bool
        Binary image indicating the valid, non-NaN, regions in the original
        image.
    """
    mask = ~np.isnan(img)

    if not mask.any():
        warnings.warn("All elements are NaN, no filling possible")
        return img.copy(), mask

    newimg = img.copy()
    if mask.all():
        return newimg, mask

    # Use distance transform to find nearest non-NaN pixel for each NaN pixel
    _, indices = distance_transform_edt(~mask, return_distances=True, return_indices=True)
    newimg = img[tuple(indices)]

    return newimg, mask


def replacenan(img, defaultval=0.0):
    """Replace NaNs in an array with a specified value.

    Parameters
    ----------
    img : ndarray
        The array containing NaN values.
    defaultval : float, optional
        The default value to replace NaNs. Default is 0.

    Returns
    -------
    newimg : ndarray
        Filled image.
    mask : ndarray of bool
        Boolean image indicating non-NaN regions in the original image.
    """
    mask = ~np.isnan(img)
    newimg = img.copy()
    newimg[np.isnan(img)] = defaultval
    return newimg, mask


def hysthresh(img, T1, T2):
    """Hysteresis thresholding of an image.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be thresholded.
    T1, T2 : float
        Upper and lower threshold values. T1 and T2 can be entered in any
        order, the larger of the two values is used as the upper threshold.

    Returns
    -------
    bw : ndarray of bool
        The binary thresholded image.

    Notes
    -----
    All pixels with values above threshold T1 are marked as edges. All pixels
    that are connected to points that have been marked as edges and with values
    above threshold T2 are also marked as edges. Eight connectivity is used.
    """
    if T1 < T2:
        T1, T2 = T2, T1

    bw = np.zeros(img.shape, dtype=bool)

    # Form 8-connected components of pixels with value >= lower threshold
    # Julia uses strel_box((3,3)) which is 8-connectivity; scipy.ndimage.label
    # defaults to 4-connectivity (cross), so we must pass a 3x3 ones structure.
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(img >= T2, structure=structure)

    # For each component, check if any pixel exceeds T1
    for n in range(1, num_features + 1):
        component = labeled == n
        if np.any(img[component] >= T1):
            bw[component] = True

    return bw


def imgnormalise(img, reqmean=None, reqvar=None):
    """Normalise image values to 0-1, or to desired mean and variance.

    Parameters
    ----------
    img : ndarray
        A grey-level input image.
    reqmean : float, optional
        The required mean value of the image.
    reqvar : float, optional
        The required variance of the image.

    Returns
    -------
    nimg : ndarray
        Normalised image.

    Notes
    -----
    If reqmean and reqvar are not provided, offsets and rescales image so that
    the minimum value is 0 and the maximum value is 1.
    If reqmean and reqvar are provided, offsets and rescales image so that
    nimg has mean reqmean and variance reqvar.
    """
    if reqmean is not None and reqvar is not None:
        if np.size(img) <= 1:
            return np.full_like(img, float(reqmean), dtype=np.float64)
        std = np.std(img, ddof=1)
        if not np.isfinite(std) or std <= np.finfo(float).eps:
            return np.full_like(img, float(reqmean), dtype=np.float64)
        n = img - np.mean(img)
        n = n / std
        return reqmean + n * np.sqrt(reqvar)
    else:
        n = img - np.min(img)
        mx = np.max(n)
        if mx > 0:
            n = n / mx
        return n


# Alias
imgnormalize = imgnormalise


def histtruncate(img, lHistCut, uHistCut=None):
    """Truncate ends of an image histogram.

    Function truncates a specified percentage of the lower and upper ends
    of an image histogram. This operation allows grey levels to be
    distributed across the primary part of the histogram.

    Parameters
    ----------
    img : ndarray (2D)
        Image to be processed.
    lHistCut : float
        Percentage of the lower end of the histogram to saturate.
    uHistCut : float, optional
        Percentage of the upper end of the histogram to saturate.
        If not provided, defaults to the value of lHistCut.

    Returns
    -------
    newimg : ndarray
        Image with values clipped at the specified histogram fraction values.
    """
    if uHistCut is None:
        uHistCut = lHistCut

    if lHistCut < 0 or lHistCut > 100 or uHistCut < 0 or uHistCut > 100:
        raise ValueError("Histogram truncation values must be between 0 and 100")

    if img.ndim > 2:
        raise ValueError("histtruncate only defined for grey scale images")

    newimg = img.copy()
    sortv = np.sort(newimg.ravel())

    # Ignore NaN values
    N = int(np.sum(~np.isnan(sortv)))

    # Compute indices corresponding to specified upper and lower fractions
    lind = int(np.floor(N * lHistCut / 100))
    hind = int(np.ceil(N - N * uHistCut / 100)) - 1

    lind = max(lind, 0)
    hind = min(hind, N - 1)

    low_val = sortv[lind]
    high_val = sortv[hind]

    newimg[newimg < low_val] = low_val
    newimg[newimg > high_val] = high_val

    return newimg
