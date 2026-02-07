ImagePhaseCongruency
=======================

----------------------------------------------

![banner image](logo.png)

## Installation

```bash
pip install .
```

Or for development:

```bash
pip install -e ".[dev]"
```

### Dependencies

- NumPy (>= 1.20)
- SciPy (>= 1.7)

## Summary

This package provides a collection of image processing functions that exploit
the importance of phase information in our perception of images.  Local phase
information, rather than local image gradients, is used as the fundamental
building block for constructing feature detectors.

This is a Python port of the Julia package
[ImagePhaseCongruency.jl](https://github.com/peterkovesi/ImagePhaseCongruency.jl)
by Peter Kovesi.

The functions form two main groups:

1) Functions that detect specific patterns of local phase for the purpose of feature detection. These include functions for the detection of edges, lines and corner features, and functions for detecting local symmetry.

2) Functions that enhance an image in a way that does not corrupt the local phase so that our perception of important features are not disrupted.  These include functions for dynamic range compression and for denoising.


## Quick Start

```python
import numpy as np
from phasecongruency import phasecongmono, phasecong3, step2line

# Generate a test image
img = step2line(512)

# Phase congruency using monogenic filters (fast)
PC, orientation, phase_type, T = phasecongmono(img)

# Phase congruency using log-Gabor filters (oriented output)
M, m, orientation, feat_type, EO, T = phasecong3(img)
```


## Function Reference

### Feature Detection (Phase Congruency)

| Function | Description |
|---|---|
| `phasecongmono(img, ...)` | Phase congruency using monogenic filters |
| `phasecong3(img, ...)` | Edge and corner phase congruency via log-Gabor filters |
| `phasesymmono(img, ...)` | Phase symmetry using monogenic filters |
| `phasesym(img, ...)` | Phase symmetry via log-Gabor filters |

### Image Enhancement

| Function | Description |
|---|---|
| `ppdrc(img, wavelength, ...)` | Phase preserving dynamic range compression |
| `ppdenoise(img, ...)` | Phase preserving wavelet denoising |

### Filtering

| Function | Description |
|---|---|
| `gaborconvolve(img, ...)` | Convolve with log-Gabor filter bank |
| `monofilt(img, ...)` | Apply monogenic filters |
| `highpassmonogenic(img, ...)` | Highpass monogenic filtering |
| `bandpassmonogenic(img, ...)` | Bandpass monogenic filtering |

### Frequency Domain Filters

| Function | Description |
|---|---|
| `filtergrids(rows, cols)` | Generate frequency domain grids |
| `filtergrid(rows, cols)` | Generate frequency radius grid |
| `lowpassfilter(sze, cutoff, n)` | Butterworth low-pass filter |
| `highpassfilter(sze, cutoff, n)` | Butterworth high-pass filter |
| `bandpassfilter(sze, cutin, cutoff, n)` | Butterworth band-pass filter |
| `highboostfilter(sze, cutoff, n, boost)` | Butterworth high-boost filter |
| `loggabor(f, fo, sigmaOnf)` | Log-Gabor function |
| `monogenicfilters(rows, cols)` | Monogenic filter pair |
| `packedmonogenicfilters(rows, cols)` | Packed monogenic filters |
| `cosineangularfilter(...)` | Cosine angular filter |
| `gaussianangularfilter(...)` | Gaussian angular filter |
| `gridangles(freq, fx, fy)` | Filter grid angles |
| `perfft2(img)` | Periodic FFT (Moisan decomposition) |
| `geoseries(s1, mult, n)` | Generate geometric series |

### Synthetic Test Images

| Function | Description |
|---|---|
| `step2line(sze, ...)` | Step to line interpolation image |
| `circsine(sze, ...)` | Circular sine wave grating |
| `starsine(sze, ...)` | Star sine wave grating |
| `noiseonf(sze, p)` | 1/f^p spectrum noise |
| `nophase(img)` | Randomize image phase |
| `quantizephase(img, N)` | Quantize image phase |
| `swapphase(img1, img2)` | Swap phase between images |

### Utilities

| Function | Description |
|---|---|
| `replacenan(img, val)` | Replace NaN values |
| `fillnan(img)` | Fill NaN with nearest valid value |
| `hysthresh(img, T1, T2)` | Hysteresis thresholding |
| `imgnormalize(img, ...)` | Normalize image values |
| `histtruncate(img, lcut, ucut)` | Truncate histogram ends |

## Testing

```bash
pytest tests/ -v
```

## Original Documentation

- [Julia package documentation](https://peterkovesi.github.io/ImagePhaseCongruency.jl/dev/index.html)
- [Examples](https://peterkovesi.github.io/ImagePhaseCongruency.jl/dev/examples/)
- [Function reference](https://peterkovesi.github.io/ImagePhaseCongruency.jl/dev/functions/)
