"""
Image Phase Congruency

Phase based feature detection and image enhancement.

Peter Kovesi
peterkovesi.com
"""

# Phase congruency and feature detection
from .phasecongruency import (
    phasecongmono,
    phasesymmono,
    ppdrc,
    highpassmonogenic,
    bandpassmonogenic,
    gaborconvolve,
    monofilt,
    phasecong3,
    phasesym,
    ppdenoise,
)

# Frequency domain filters
from .frequencyfilt import (
    filtergrid,
    filtergrids,
    gridangles,
    cosineangularfilter,
    gaussianangularfilter,
    lowpassfilter,
    highpassfilter,
    bandpassfilter,
    highboostfilter,
    loggabor,
    monogenicfilters,
    packedmonogenicfilters,
    perfft2,
    geoseries,
)

# Synthetic test images
from .syntheticimages import (
    step2line,
    circsine,
    starsine,
    noiseonf,
    nophase,
    quantizephase,
    swapphase,
)

# Utilities
from .utilities import (
    replacenan,
    fillnan,
    hysthresh,
    imgnormalise,
    imgnormalize,
    histtruncate,
)

__version__ = "1.0.3"

__all__ = [
    # Phase congruency
    "phasecongmono",
    "phasesymmono",
    "ppdrc",
    "highpassmonogenic",
    "bandpassmonogenic",
    "gaborconvolve",
    "monofilt",
    "phasecong3",
    "phasesym",
    "ppdenoise",
    # Frequency filters
    "filtergrid",
    "filtergrids",
    "gridangles",
    "cosineangularfilter",
    "gaussianangularfilter",
    "lowpassfilter",
    "highpassfilter",
    "bandpassfilter",
    "highboostfilter",
    "loggabor",
    "monogenicfilters",
    "packedmonogenicfilters",
    "perfft2",
    "geoseries",
    # Synthetic images
    "step2line",
    "circsine",
    "starsine",
    "noiseonf",
    "nophase",
    "quantizephase",
    "swapphase",
    # Utilities
    "replacenan",
    "fillnan",
    "hysthresh",
    "imgnormalise",
    "imgnormalize",
    "histtruncate",
]
