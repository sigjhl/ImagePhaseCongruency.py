"""
Functions related to the phase congruency model of feature perception
and phase based approaches to image processing.

Copyright (c) 2015-2018 Peter Kovesi
peterkovesi.com

MIT License.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

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


_VALID_BACKENDS = {"numpy", "torch", "torch-mps", "auto"}
_AUTO_BACKEND_CACHE_VERSION = 1
_AUTO_CACHE_ENV_VAR = "PHASECONGRUENCY_AUTO_CACHE_PATH"
_AUTO_CACHE_PATH = None
_AUTO_BACKEND_CACHE = None
_AUTO_SELECT_MARGIN = 0.95


def _resolve_backend(backend):
    """Normalise backend name and validate it."""
    if backend is None:
        backend = "numpy"
    backend = str(backend).lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported backends: "
            f"{', '.join(sorted(_VALID_BACKENDS))}"
        )
    return backend


def _default_auto_cache_path():
    """Resolve persistent cache path for auto-backend decisions."""
    override = os.environ.get(_AUTO_CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser()

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "phasecongruency" / "auto_backend_cache.json"

    return Path.home() / ".cache" / "phasecongruency" / "auto_backend_cache.json"


def _auto_cache_path():
    """Return auto-backend cache path (memoized)."""
    global _AUTO_CACHE_PATH
    if _AUTO_CACHE_PATH is None:
        _AUTO_CACHE_PATH = _default_auto_cache_path()
    return _AUTO_CACHE_PATH


def _load_auto_backend_cache():
    """Load persistent auto-backend cache once per process."""
    global _AUTO_BACKEND_CACHE
    if _AUTO_BACKEND_CACHE is not None:
        return _AUTO_BACKEND_CACHE

    cache_path = _auto_cache_path()
    entries = {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if (
            isinstance(payload, dict)
            and payload.get("cache_version") == _AUTO_BACKEND_CACHE_VERSION
            and isinstance(payload.get("entries"), dict)
        ):
            entries = payload["entries"]
    except (OSError, json.JSONDecodeError):
        entries = {}

    _AUTO_BACKEND_CACHE = entries
    return _AUTO_BACKEND_CACHE


def _save_auto_backend_cache():
    """Persist auto-backend cache; silently ignore write failures."""
    cache = _load_auto_backend_cache()
    cache_path = _auto_cache_path()
    payload = {
        "cache_version": _AUTO_BACKEND_CACHE_VERSION,
        "entries": cache,
    }

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
        tmp_path.replace(cache_path)
    except OSError:
        return


def _runtime_signature(torch):
    """Signature used to invalidate cache across environment changes."""
    return {
        "cache_version": _AUTO_BACKEND_CACHE_VERSION,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def _auto_cache_key(function_name, img_shape, params, torch_device, torch):
    """Create a stable cache key for backend auto-selection."""
    key_payload = {
        "function": function_name,
        "shape": [int(img_shape[0]), int(img_shape[1])],
        "params": params,
        "torch_device": str(torch_device),
        "runtime": _runtime_signature(torch),
    }
    return json.dumps(key_payload, sort_keys=True, separators=(",", ":"))


def _import_torch():
    """Import torch lazily so NumPy users avoid hard dependency on torch."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Torch backend requested but PyTorch is not installed. "
            "Install it with `pip install phasecongruency[gpu]` or install "
            "a PyTorch build that supports your accelerator."
        ) from exc
    return torch


def _select_torch_device(torch, backend, device):
    """Select torch device for requested backend."""
    if device is not None:
        return torch.device(device)

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    cuda_available = bool(torch.cuda.is_available())

    if backend == "torch-mps":
        if not mps_available:
            raise RuntimeError(
                "backend='torch-mps' requested, but torch MPS is not available."
            )
        return torch.device("mps")

    if backend == "auto":
        if mps_available:
            return torch.device("mps")
        if cuda_available:
            return torch.device("cuda")
        return torch.device("cpu")

    # backend == "torch"
    if mps_available:
        return torch.device("mps")
    if cuda_available:
        return torch.device("cuda")
    return torch.device("cpu")


def _torch_dtypes(torch, device):
    """Choose dtypes compatible with selected torch device."""
    if str(device).startswith("mps"):
        return torch.float32, torch.complex64
    return torch.float64, torch.complex128


def _synchronize_torch(torch, device):
    """Synchronize torch device so timing includes all queued work."""
    device_type = str(device).split(":")[0]
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _benchmark_callable(fn, torch=None, device=None):
    """Benchmark a single callable invocation."""
    if torch is not None and device is not None:
        _synchronize_torch(torch, device)
    t0 = time.perf_counter()
    result = fn()
    if torch is not None and device is not None:
        _synchronize_torch(torch, device)
    return time.perf_counter() - t0, result


def _run_auto_backend(
    *,
    function_name,
    img_shape,
    params,
    requested_device,
    numpy_call,
    torch_call,
):
    """Run function using cached adaptive backend decision."""
    torch = _import_torch()
    torch_device = _select_torch_device(torch, "auto", requested_device)
    torch_backend = "torch-mps" if str(torch_device).startswith("mps") else "torch"

    cache_key = _auto_cache_key(function_name, img_shape, params, torch_device, torch)
    cache = _load_auto_backend_cache()
    entry = cache.get(cache_key)
    if isinstance(entry, dict):
        selected = entry.get("selected")
        if selected == "numpy":
            return numpy_call()
        if selected == "torch":
            try:
                return torch_call(torch_backend, str(torch_device))
            except Exception:
                cache[cache_key] = {
                    "selected": "numpy",
                    "fallback_reason": "cached_torch_failed",
                    "measured_at_epoch": time.time(),
                }
                _save_auto_backend_cache()
                return numpy_call()

    np_time, np_result = _benchmark_callable(numpy_call)
    try:
        torch_time, torch_result = _benchmark_callable(
            lambda: torch_call(torch_backend, str(torch_device)),
            torch=torch,
            device=torch_device,
        )
    except Exception as exc:
        cache[cache_key] = {
            "selected": "numpy",
            "numpy_time_sec": np_time,
            "torch_error": repr(exc),
            "torch_backend": torch_backend,
            "measured_at_epoch": time.time(),
        }
        _save_auto_backend_cache()
        return np_result

    selected = "torch" if torch_time < np_time * _AUTO_SELECT_MARGIN else "numpy"
    cache[cache_key] = {
        "selected": selected,
        "numpy_time_sec": np_time,
        "torch_time_sec": torch_time,
        "torch_backend": torch_backend,
        "measured_at_epoch": time.time(),
    }
    _save_auto_backend_cache()

    if selected == "torch":
        return torch_result
    return np_result


def _to_numpy(tensor):
    """Convert torch tensor to NumPy array."""
    return tensor.detach().cpu().numpy()


def _torch_lowpassfilter(freq, cutoff, n):
    """Torch equivalent of Butterworth low-pass filter over a frequency grid."""
    return 1.0 / (1.0 + (freq / cutoff) ** (2 * n))


def _torch_loggabor(freq, fo, sigmaonf, eps):
    """Torch equivalent of loggabor() supporting tensor frequency grids."""
    out = freq.new_zeros(freq.shape)
    mask = freq >= eps
    if bool(mask.any()):
        log_sigma_sq = np.log(sigmaonf) ** 2
        out[mask] = ((-(freq[mask] / fo).log() ** 2) / (2 * log_sigma_sq)).exp()
    return out


def _rayleighmode_torch(X, nbins=50):
    """Torch-friendly Rayleigh mode estimate via NumPy fallback."""
    return _rayleighmode(_to_numpy(X), nbins=nbins)


def _torch_prepare_image(img, backend, device):
    """Prepare a 2D image tensor and FFT for torch backends."""
    torch = _import_torch()
    backend = _resolve_backend(backend)
    tdevice = _select_torch_device(torch, backend, device)
    real_dtype, complex_dtype = _torch_dtypes(torch, tdevice)

    img_np = np.asarray(img, dtype=np.float64)
    if img_np.ndim != 2:
        raise ValueError("img must be a 2D array")

    img_t = torch.as_tensor(img_np, dtype=real_dtype, device=tdevice)
    IMG = torch.fft.fft2(img_t)
    return torch, tdevice, real_dtype, complex_dtype, img_t, IMG


def _torch_cosineangularfilter(angl, wavelen, sintheta, costheta):
    """Torch equivalent of cosineangularfilter()."""
    sinangl = np.sin(angl)
    cosangl = np.cos(angl)
    ds = sintheta * cosangl - costheta * sinangl
    dc = costheta * cosangl + sintheta * sinangl
    dtheta = ds.atan2(dc).abs()
    dtheta = (dtheta * (2 * np.pi / wavelen)).clamp(max=np.pi)
    return (dtheta.cos() + 1) / 2


def _torch_gaussianangularfilter(angl, thetaSigma, sintheta, costheta):
    """Torch equivalent of gaussianangularfilter()."""
    sinangl = np.sin(angl)
    cosangl = np.cos(angl)
    ds = sintheta * cosangl - costheta * sinangl
    dc = costheta * cosangl + sintheta * sinangl
    dtheta = ds.atan2(dc)
    return ((-dtheta ** 2) / (2 * thetaSigma ** 2)).exp()


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
    Scaling affects the results. Inputs with very small magnitudes may
    benefit from rescaling before processing.
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
            if dimg[k].size >= 1:
                dimg[k][np.unravel_index(0, dimg[k].shape, order="F")] = maxrange
            if dimg[k].size >= 2:
                dimg[k][np.unravel_index(1, dimg[k].shape, order="F")] = -maxrange

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
                  deviationgain=1.5, backend="numpy", device=None):
    """Phase congruency of an image using monogenic filters.

    This variant is typically faster than phasecong3() and returns a reduced
    output set.

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
    backend : {"numpy", "torch", "torch-mps", "auto"}, optional
        Compute backend. ``"numpy"`` keeps existing behavior.
        ``"auto"`` prefers MPS on Apple Silicon, then CUDA, then CPU.
    device : str, optional
        Explicit torch device string (for example ``"mps"``, ``"cuda"``,
        or ``"cpu"``) when using a torch backend.

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
    backend = _resolve_backend(backend)
    if backend in {"torch", "torch-mps"}:
        return _phasecongmono_torch(
            img, nscale=nscale, minwavelength=minwavelength, mult=mult,
            sigmaonf=sigmaonf, k=k, noisemethod=noisemethod, cutoff=cutoff,
            g=g, deviationgain=deviationgain, backend=backend, device=device,
        )
    if backend == "auto":
        try:
            _import_torch()
        except ImportError:
            backend = "numpy"
        else:
            img_shape = np.asarray(img).shape
            if len(img_shape) != 2:
                return phasecongmono(
                    img, nscale=nscale, minwavelength=minwavelength, mult=mult,
                    sigmaonf=sigmaonf, k=k, noisemethod=noisemethod,
                    cutoff=cutoff, g=g, deviationgain=deviationgain,
                    backend="numpy", device=device,
                )
            return _run_auto_backend(
                function_name="phasecongmono",
                img_shape=img_shape,
                params={
                    "nscale": nscale,
                    "minwavelength": minwavelength,
                    "mult": mult,
                    "sigmaonf": sigmaonf,
                    "k": k,
                    "noisemethod": noisemethod,
                    "cutoff": cutoff,
                    "g": g,
                    "deviationgain": deviationgain,
                },
                requested_device=device,
                numpy_call=lambda: phasecongmono(
                    img, nscale=nscale, minwavelength=minwavelength, mult=mult,
                    sigmaonf=sigmaonf, k=k, noisemethod=noisemethod,
                    cutoff=cutoff, g=g, deviationgain=deviationgain,
                    backend="numpy", device=device,
                ),
                torch_call=lambda b, d: phasecongmono(
                    img, nscale=nscale, minwavelength=minwavelength, mult=mult,
                    sigmaonf=sigmaonf, k=k, noisemethod=noisemethod,
                    cutoff=cutoff, g=g, deviationgain=deviationgain,
                    backend=b, device=d,
                ),
            )

    img = np.asarray(img, dtype=np.float64)
    if nscale < 2:
        raise ValueError("nscale must be at least 2")
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
    lowpass_45 = lowpassfilter(freq, 0.45, 15)

    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        # Construct bandpassed image in frequency domain.
        IMGF = IMG * loggabor(freq, fo, sigmaonf) * lowpass_45

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
                 k=2.0, polarity=0, noisemethod=-1, backend="numpy",
                 device=None):
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
    backend : {"numpy", "torch", "torch-mps", "auto"}, optional
        Compute backend. ``"numpy"`` keeps existing behavior.
        ``"auto"`` prefers MPS on Apple Silicon, then CUDA, then CPU.
    device : str, optional
        Explicit torch device string (for example ``"mps"``, ``"cuda"``,
        or ``"cpu"``) when using a torch backend.

    Returns
    -------
    phSym : ndarray
        Phase symmetry image (values between 0 and 1).
    symmetryEnergy : ndarray
        Un-normalised raw symmetry energy.
    T : float
        Calculated noise threshold.
    """
    backend = _resolve_backend(backend)
    if backend in {"torch", "torch-mps"}:
        return _phasesymmono_torch(
            img, nscale=nscale, minwavelength=minwavelength, mult=mult,
            sigmaonf=sigmaonf, k=k, polarity=polarity, noisemethod=noisemethod,
            backend=backend, device=device,
        )
    if backend == "auto":
        try:
            _import_torch()
        except ImportError:
            backend = "numpy"
        else:
            img_shape = np.asarray(img).shape
            if len(img_shape) != 2:
                return phasesymmono(
                    img, nscale=nscale, minwavelength=minwavelength, mult=mult,
                    sigmaonf=sigmaonf, k=k, polarity=polarity,
                    noisemethod=noisemethod, backend="numpy", device=device,
                )
            return _run_auto_backend(
                function_name="phasesymmono",
                img_shape=img_shape,
                params={
                    "nscale": nscale,
                    "minwavelength": minwavelength,
                    "mult": mult,
                    "sigmaonf": sigmaonf,
                    "k": k,
                    "polarity": polarity,
                    "noisemethod": noisemethod,
                },
                requested_device=device,
                numpy_call=lambda: phasesymmono(
                    img, nscale=nscale, minwavelength=minwavelength, mult=mult,
                    sigmaonf=sigmaonf, k=k, polarity=polarity,
                    noisemethod=noisemethod, backend="numpy", device=device,
                ),
                torch_call=lambda b, d: phasesymmono(
                    img, nscale=nscale, minwavelength=minwavelength, mult=mult,
                    sigmaonf=sigmaonf, k=k, polarity=polarity,
                    noisemethod=noisemethod, backend=b, device=d,
                ),
            )

    img = np.asarray(img, dtype=np.float64)
    epsilon = 0.0001

    rows, cols = img.shape
    IMG = fft2(img)

    tau = 0.0
    symmetryEnergy = np.zeros((rows, cols))
    sumAn = np.zeros((rows, cols))

    H, freq = packedmonogenicfilters(rows, cols)
    lowpass_40 = lowpassfilter(freq, 0.4, 10)

    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        IMGF = IMG * loggabor(freq, fo, sigmaonf) * lowpass_40

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

        logGabor = loggabor(freq, fo, sigmaOnf)

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
    lowpass_45 = lowpassfilter(freq, 0.45, 15)
    for s in range(nscale):
        wavelength = minWaveLength * mult**s
        fo = 1.0 / wavelength

        lg = loggabor(freq, fo, sigmaOnf) * lowpass_45

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
               sigmaonf=0.55, k=2.0, cutoff=0.5, g=10.0, noisemethod=-1,
               backend="numpy", device=None, return_eo=True):
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
    backend : {"numpy", "torch", "torch-mps", "auto"}, optional
        Compute backend. ``"numpy"`` keeps existing behavior.
        ``"auto"`` prefers MPS on Apple Silicon, then CUDA, then CPU.
    device : str, optional
        Explicit torch device string (for example ``"mps"``, ``"cuda"``,
        or ``"cpu"``) when using a torch backend.
    return_eo : bool, optional
        If True (default), return the full complex EO bank as in the
        historical API. If False, skips EO bank materialization to reduce
        memory usage and returns ``None`` for EO.

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
        2D array of complex valued convolution results, or ``None`` when
        ``return_eo=False``.
    T : float
        Calculated noise threshold.
    """
    backend = _resolve_backend(backend)
    if backend in {"torch", "torch-mps"}:
        return _phasecong3_torch(
            img, nscale=nscale, norient=norient, minwavelength=minwavelength,
            mult=mult, sigmaonf=sigmaonf, k=k, cutoff=cutoff, g=g,
            noisemethod=noisemethod, backend=backend, device=device,
            return_eo=return_eo,
        )
    if backend == "auto":
        try:
            _import_torch()
        except ImportError:
            backend = "numpy"
        else:
            img_shape = np.asarray(img).shape
            if len(img_shape) != 2:
                return phasecong3(
                    img, nscale=nscale, norient=norient, minwavelength=minwavelength,
                    mult=mult, sigmaonf=sigmaonf, k=k, cutoff=cutoff, g=g,
                    noisemethod=noisemethod, backend="numpy", device=device,
                    return_eo=return_eo,
                )
            return _run_auto_backend(
                function_name="phasecong3",
                img_shape=img_shape,
                params={
                    "nscale": nscale,
                    "norient": norient,
                    "minwavelength": minwavelength,
                    "mult": mult,
                    "sigmaonf": sigmaonf,
                    "k": k,
                    "cutoff": cutoff,
                    "g": g,
                    "noisemethod": noisemethod,
                    "return_eo": return_eo,
                },
                requested_device=device,
                numpy_call=lambda: phasecong3(
                    img, nscale=nscale, norient=norient, minwavelength=minwavelength,
                    mult=mult, sigmaonf=sigmaonf, k=k, cutoff=cutoff, g=g,
                    noisemethod=noisemethod, backend="numpy", device=device,
                    return_eo=return_eo,
                ),
                torch_call=lambda b, d: phasecong3(
                    img, nscale=nscale, norient=norient, minwavelength=minwavelength,
                    mult=mult, sigmaonf=sigmaonf, k=k, cutoff=cutoff, g=g,
                    noisemethod=noisemethod, backend=b, device=d,
                    return_eo=return_eo,
                ),
            )

    img = np.asarray(img, dtype=np.float64)
    if nscale < 2:
        raise ValueError("nscale must be at least 2")
    epsilon = 1e-5
    rows, cols = img.shape
    IMG = fft2(img)

    logGabor_arr = [None] * nscale
    EO = [[None] * norient for _ in range(nscale)] if return_eo else None
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
    lowpass_45 = lowpassfilter(freq, 0.45, 15)
    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        lg = loggabor(freq, fo, sigmaonf) * lowpass_45
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
        eo_this_orient = [None] * nscale

        for s in range(nscale):
            filt = logGabor_arr[s] * angfilter
            eo = ifft2(IMG * filt)
            eo_this_orient[s] = eo
            if return_eo:
                EO[s][o] = eo

            An[:] = np.abs(eo)
            sumAn_ThisOrient += An
            sumE_ThisOrient += np.real(eo)
            sumO_ThisOrient += np.imag(eo)

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
        for eo in eo_this_orient:
            E_val = np.real(eo)
            O_val = np.imag(eo)
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
             sigmaonf=0.55, k=2.0, polarity=0, noisemethod=-1,
             backend="numpy", device=None):
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
    backend : {"numpy", "torch", "torch-mps", "auto"}, optional
        Compute backend. ``"numpy"`` keeps existing behavior.
        ``"auto"`` prefers MPS on Apple Silicon, then CUDA, then CPU.
    device : str, optional
        Explicit torch device string (for example ``"mps"``, ``"cuda"``,
        or ``"cpu"``) when using a torch backend.

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
    backend = _resolve_backend(backend)
    if backend in {"torch", "torch-mps"}:
        return _phasesym_torch(
            img, nscale=nscale, norient=norient, minwavelength=minwavelength,
            mult=mult, sigmaonf=sigmaonf, k=k, polarity=polarity,
            noisemethod=noisemethod, backend=backend, device=device,
        )
    if backend == "auto":
        try:
            _import_torch()
        except ImportError:
            backend = "numpy"
        else:
            img_shape = np.asarray(img).shape
            if len(img_shape) != 2:
                return phasesym(
                    img, nscale=nscale, norient=norient, minwavelength=minwavelength,
                    mult=mult, sigmaonf=sigmaonf, k=k, polarity=polarity,
                    noisemethod=noisemethod, backend="numpy", device=device,
                )
            return _run_auto_backend(
                function_name="phasesym",
                img_shape=img_shape,
                params={
                    "nscale": nscale,
                    "norient": norient,
                    "minwavelength": minwavelength,
                    "mult": mult,
                    "sigmaonf": sigmaonf,
                    "k": k,
                    "polarity": polarity,
                    "noisemethod": noisemethod,
                },
                requested_device=device,
                numpy_call=lambda: phasesym(
                    img, nscale=nscale, norient=norient, minwavelength=minwavelength,
                    mult=mult, sigmaonf=sigmaonf, k=k, polarity=polarity,
                    noisemethod=noisemethod, backend="numpy", device=device,
                ),
                torch_call=lambda b, d: phasesym(
                    img, nscale=nscale, norient=norient, minwavelength=minwavelength,
                    mult=mult, sigmaonf=sigmaonf, k=k, polarity=polarity,
                    noisemethod=noisemethod, backend=b, device=d,
                ),
            )

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
    lowpass_45 = lowpassfilter(freq, 0.45, 15)
    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        lg = loggabor(freq, fo, sigmaonf) * lowpass_45
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
              sigmaonf=0.55, dthetaonsigma=1.0, k=3.0, softness=1.0,
              backend="numpy", device=None):
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
    backend : {"numpy", "torch", "torch-mps", "auto"}, optional
        Compute backend. ``"numpy"`` keeps existing behavior.
        ``"auto"`` prefers MPS on Apple Silicon, then CUDA, then CPU.
    device : str, optional
        Explicit torch device string (for example ``"mps"``, ``"cuda"``,
        or ``"cpu"``) when using a torch backend.

    Returns
    -------
    cleanimage : ndarray
        Denoised image.
    """
    backend = _resolve_backend(backend)
    if backend in {"torch", "torch-mps"}:
        return _ppdenoise_torch(
            img, nscale=nscale, norient=norient, mult=mult,
            minwavelength=minwavelength, sigmaonf=sigmaonf,
            dthetaonsigma=dthetaonsigma, k=k, softness=softness,
            backend=backend, device=device,
        )
    if backend == "auto":
        try:
            _import_torch()
        except ImportError:
            backend = "numpy"
        else:
            img_shape = np.asarray(img).shape
            if len(img_shape) != 2:
                return ppdenoise(
                    img, nscale=nscale, norient=norient, mult=mult,
                    minwavelength=minwavelength, sigmaonf=sigmaonf,
                    dthetaonsigma=dthetaonsigma, k=k, softness=softness,
                    backend="numpy", device=device,
                )
            return _run_auto_backend(
                function_name="ppdenoise",
                img_shape=img_shape,
                params={
                    "nscale": nscale,
                    "norient": norient,
                    "mult": mult,
                    "minwavelength": minwavelength,
                    "sigmaonf": sigmaonf,
                    "dthetaonsigma": dthetaonsigma,
                    "k": k,
                    "softness": softness,
                },
                requested_device=device,
                numpy_call=lambda: ppdenoise(
                    img, nscale=nscale, norient=norient, mult=mult,
                    minwavelength=minwavelength, sigmaonf=sigmaonf,
                    dthetaonsigma=dthetaonsigma, k=k, softness=softness,
                    backend="numpy", device=device,
                ),
                torch_call=lambda b, d: ppdenoise(
                    img, nscale=nscale, norient=norient, mult=mult,
                    minwavelength=minwavelength, sigmaonf=sigmaonf,
                    dthetaonsigma=dthetaonsigma, k=k, softness=softness,
                    backend=b, device=d,
                ),
            )

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
            filt = loggabor(freq, fo, sigmaonf) * angfilter

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


def _phasecongmono_torch(
    img, nscale=4, minwavelength=3, mult=2.1, sigmaonf=0.55,
    k=3.0, noisemethod=-1, cutoff=0.5, g=10.0, deviationgain=1.5,
    backend="auto", device=None,
):
    """Torch backend for phasecongmono()."""
    if nscale < 2:
        raise ValueError("nscale must be at least 2")

    epsilon = 0.0001
    torch, tdevice, real_dtype, complex_dtype, img_t, IMG = _torch_prepare_image(
        img, backend, device
    )
    rows, cols = img_t.shape

    H_np, freq_np = packedmonogenicfilters(rows, cols)
    H = torch.as_tensor(H_np, dtype=complex_dtype, device=tdevice)
    freq = torch.as_tensor(freq_np, dtype=real_dtype, device=tdevice)
    lowpass_45 = _torch_lowpassfilter(freq, 0.45, 15)
    eps_freq = torch.finfo(real_dtype).eps

    sumAn = torch.zeros_like(img_t)
    sumf = torch.zeros_like(img_t)
    sumh1 = torch.zeros_like(img_t)
    sumh2 = torch.zeros_like(img_t)
    maxAn = torch.zeros_like(img_t)

    tau = 0.0
    T = 0.0

    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        IMGF = IMG * _torch_loggabor(freq, fo, sigmaonf, eps_freq) * lowpass_45

        f = torch.real(torch.fft.ifft2(IMGF))
        h = torch.fft.ifft2(IMGF * H)

        h1 = torch.real(h)
        h2 = torch.imag(h)
        An = torch.sqrt(f**2 + h1**2 + h2**2)
        sumAn = sumAn + An
        sumf = sumf + f
        sumh1 = sumh1 + h1
        sumh2 = sumh2 + h2

        if s == 0:
            if abs(noisemethod + 1) < epsilon:
                tau = float(torch.median(sumAn).item() / np.sqrt(np.log(4)))
            elif abs(noisemethod + 2) < epsilon:
                tau = float(_rayleighmode_torch(sumAn))
            maxAn = An.clone()
        else:
            maxAn = torch.maximum(maxAn, An)

    width = (sumAn / (maxAn + epsilon) - 1) / (nscale - 1)
    weight = 1.0 / (1 + torch.exp((cutoff - width) * g))

    if noisemethod >= 0:
        T = float(noisemethod)
    else:
        totalTau = tau * (1 - (1 / mult)**nscale) / (1 - (1 / mult))
        EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
        EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
        T = float(EstNoiseEnergyMean + k * EstNoiseEnergySigma)

    or_ = torch.atan(-sumh2 / sumh1)
    ft = torch.atan2(sumf, torch.sqrt(sumh1**2 + sumh2**2))
    energy = torch.sqrt(sumf**2 + sumh1**2 + sumh2**2)

    PC = (
        weight
        * torch.clamp(1 - deviationgain * torch.acos(energy / (sumAn + epsilon)), min=0.0)
        * torch.clamp(energy - T, min=0.0)
        / (energy + epsilon)
    )

    return _to_numpy(PC), _to_numpy(or_), _to_numpy(ft), T


def _phasesymmono_torch(
    img, nscale=5, minwavelength=3, mult=2.1, sigmaonf=0.55,
    k=2.0, polarity=0, noisemethod=-1, backend="auto", device=None,
):
    """Torch backend for phasesymmono()."""
    epsilon = 0.0001
    torch, tdevice, real_dtype, complex_dtype, img_t, IMG = _torch_prepare_image(
        img, backend, device
    )
    rows, cols = img_t.shape

    H_np, freq_np = packedmonogenicfilters(rows, cols)
    H = torch.as_tensor(H_np, dtype=complex_dtype, device=tdevice)
    freq = torch.as_tensor(freq_np, dtype=real_dtype, device=tdevice)
    lowpass_40 = _torch_lowpassfilter(freq, 0.4, 10)
    eps_freq = torch.finfo(real_dtype).eps

    tau = 0.0
    symmetryEnergy = torch.zeros_like(img_t)
    sumAn = torch.zeros_like(img_t)

    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength

        IMGF = IMG * _torch_loggabor(freq, fo, sigmaonf, eps_freq) * lowpass_40
        f = torch.real(torch.fft.ifft2(IMGF))
        h = torch.fft.ifft2(IMGF * H)

        h1 = torch.real(h)
        h2 = torch.imag(h)
        hAmp2 = h1**2 + h2**2
        sumAn = sumAn + torch.sqrt(f**2 + hAmp2)

        if polarity == 0:
            symmetryEnergy = symmetryEnergy + torch.abs(f) - torch.sqrt(hAmp2)
        elif polarity == 1:
            symmetryEnergy = symmetryEnergy + f - torch.sqrt(hAmp2)
        elif polarity == -1:
            symmetryEnergy = symmetryEnergy - f - torch.sqrt(hAmp2)

        if s == 0:
            if abs(noisemethod + 1) < epsilon:
                tau = float(torch.median(sumAn).item() / np.sqrt(np.log(4)))
            elif abs(noisemethod + 2) < epsilon:
                tau = float(_rayleighmode_torch(sumAn))

    if noisemethod >= 0:
        T = float(noisemethod)
    else:
        totalTau = tau * (1 - (1 / mult)**nscale) / (1 - (1 / mult))
        EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
        EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
        T = float(max(EstNoiseEnergyMean + k * EstNoiseEnergySigma, epsilon))

    phSym = torch.clamp(symmetryEnergy - T, min=0.0) / (sumAn + epsilon)

    return _to_numpy(phSym), _to_numpy(symmetryEnergy), T


def _phasecong3_torch(
    img, nscale=4, norient=6, minwavelength=3, mult=2.1, sigmaonf=0.55,
    k=2.0, cutoff=0.5, g=10.0, noisemethod=-1, backend="auto", device=None,
    return_eo=True,
):
    """Torch backend for phasecong3()."""
    if nscale < 2:
        raise ValueError("nscale must be at least 2")

    epsilon = 1e-5
    torch, tdevice, real_dtype, _, img_t, IMG = _torch_prepare_image(
        img, backend, device
    )
    rows, cols = img_t.shape

    logGabor_arr = [None] * nscale
    EO = [[None] * norient for _ in range(nscale)] if return_eo else None
    EnergyV = torch.zeros((rows, cols, 3), dtype=real_dtype, device=tdevice)

    covx2 = torch.zeros_like(img_t)
    covy2 = torch.zeros_like(img_t)
    covxy = torch.zeros_like(img_t)

    sumE_ThisOrient = torch.zeros_like(img_t)
    sumO_ThisOrient = torch.zeros_like(img_t)
    sumAn_ThisOrient = torch.zeros_like(img_t)
    Energy = torch.zeros_like(img_t)
    maxAn = torch.zeros_like(img_t)

    T = 0.0
    tau = 0.0

    freq_np, fx_np, fy_np = filtergrids(rows, cols)
    sintheta_np, costheta_np = gridangles(freq_np, fx_np, fy_np)

    freq = torch.as_tensor(freq_np, dtype=real_dtype, device=tdevice)
    sintheta = torch.as_tensor(sintheta_np, dtype=real_dtype, device=tdevice)
    costheta = torch.as_tensor(costheta_np, dtype=real_dtype, device=tdevice)

    lowpass_45 = _torch_lowpassfilter(freq, 0.45, 15)
    eps_freq = torch.finfo(real_dtype).eps
    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength
        lg = _torch_loggabor(freq, fo, sigmaonf, eps_freq) * lowpass_45
        logGabor_arr[s] = lg

    for o in range(norient):
        angl = o * np.pi / norient
        wavelen = 4 * np.pi / norient
        angfilter = _torch_cosineangularfilter(angl, wavelen, sintheta, costheta)

        sumE_ThisOrient.zero_()
        sumO_ThisOrient.zero_()
        sumAn_ThisOrient.zero_()
        Energy.zero_()
        eo_this_orient = [None] * nscale

        for s in range(nscale):
            filt = logGabor_arr[s] * angfilter
            eo = torch.fft.ifft2(IMG * filt)
            eo_this_orient[s] = eo
            if return_eo:
                EO[s][o] = eo

            An = torch.abs(eo)
            sumAn_ThisOrient = sumAn_ThisOrient + An
            sumE_ThisOrient = sumE_ThisOrient + torch.real(eo)
            sumO_ThisOrient = sumO_ThisOrient + torch.imag(eo)

            if s == 0:
                if abs(noisemethod + 1) < epsilon:
                    tau = float(torch.median(sumAn_ThisOrient).item() / np.sqrt(np.log(4)))
                elif abs(noisemethod + 2) < epsilon:
                    tau = float(_rayleighmode_torch(sumAn_ThisOrient))
                maxAn = An.clone()
            else:
                maxAn = torch.maximum(maxAn, An)

        EnergyV[:, :, 0] = EnergyV[:, :, 0] + sumE_ThisOrient
        EnergyV[:, :, 1] = EnergyV[:, :, 1] + np.cos(angl) * sumO_ThisOrient
        EnergyV[:, :, 2] = EnergyV[:, :, 2] + np.sin(angl) * sumO_ThisOrient

        XEnergy = torch.sqrt(sumE_ThisOrient**2 + sumO_ThisOrient**2) + epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy

        for eo in eo_this_orient:
            E_val = torch.real(eo)
            O_val = torch.imag(eo)
            Energy = Energy + (
                E_val * MeanE + O_val * MeanO
                - torch.abs(E_val * MeanO - O_val * MeanE)
            )

        if noisemethod >= 0:
            T = float(noisemethod)
        else:
            totalTau = tau * (1 - (1 / mult)**nscale) / (1 - (1 / mult))
            EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
            EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
            T = float(EstNoiseEnergyMean + k * EstNoiseEnergySigma)

        Energy = torch.clamp(Energy - T, min=0.0)

        width = (sumAn_ThisOrient / (maxAn + epsilon) - 1) / (nscale - 1)
        weight_arr = 1.0 / (1 + torch.exp((cutoff - width) * g))
        PCo = weight_arr * Energy / sumAn_ThisOrient

        covx = PCo * np.cos(angl)
        covy = PCo * np.sin(angl)
        covx2 = covx2 + covx**2
        covy2 = covy2 + covy**2
        covxy = covxy + covx * covy

    covx2 = covx2 / (norient / 2)
    covy2 = covy2 / (norient / 2)
    covxy = covxy * (4 / norient)

    denom = torch.sqrt(covxy**2 + (covx2 - covy2)**2) + epsilon
    M = (covy2 + covx2 + denom) / 2
    m = (covy2 + covx2 - denom) / 2

    or_ = torch.atan(-EnergyV[:, :, 2] / EnergyV[:, :, 1])
    OddV = torch.sqrt(EnergyV[:, :, 1]**2 + EnergyV[:, :, 2]**2)
    featType = torch.atan2(EnergyV[:, :, 0], OddV)

    EO_np = None
    if return_eo:
        EO_np = [[_to_numpy(EO[s][o]) for o in range(norient)] for s in range(nscale)]
    return _to_numpy(M), _to_numpy(m), _to_numpy(or_), _to_numpy(featType), EO_np, T


def _phasesym_torch(
    img, nscale=5, norient=6, minwavelength=3, mult=2.1, sigmaonf=0.55,
    k=2.0, polarity=0, noisemethod=-1, backend="auto", device=None,
):
    """Torch backend for phasesym()."""
    epsilon = 1e-4
    torch, tdevice, real_dtype, _, img_t, IMG = _torch_prepare_image(
        img, backend, device
    )
    rows, cols = img_t.shape

    logGabor_arr = [None] * nscale
    totalEnergy = torch.zeros_like(img_t)
    totalSumAn = torch.zeros_like(img_t)
    orientation = torch.zeros_like(img_t)
    maxEnergy = torch.zeros_like(img_t)

    tau = 0.0
    T = 0.0

    freq_np, fx_np, fy_np = filtergrids(rows, cols)
    sintheta_np, costheta_np = gridangles(freq_np, fx_np, fy_np)

    freq = torch.as_tensor(freq_np, dtype=real_dtype, device=tdevice)
    sintheta = torch.as_tensor(sintheta_np, dtype=real_dtype, device=tdevice)
    costheta = torch.as_tensor(costheta_np, dtype=real_dtype, device=tdevice)

    lowpass_45 = _torch_lowpassfilter(freq, 0.45, 15)
    eps_freq = torch.finfo(real_dtype).eps
    for s in range(nscale):
        wavelength = minwavelength * mult**s
        fo = 1.0 / wavelength
        logGabor_arr[s] = _torch_loggabor(freq, fo, sigmaonf, eps_freq) * lowpass_45

    for o in range(norient):
        angl = o * np.pi / norient
        wavelen = 4 * np.pi / norient
        angfilter = _torch_cosineangularfilter(angl, wavelen, sintheta, costheta)

        sumAn_ThisOrient = torch.zeros_like(img_t)
        Energy_ThisOrient = torch.zeros_like(img_t)

        for s in range(nscale):
            filt = logGabor_arr[s] * angfilter
            EO = torch.fft.ifft2(IMG * filt)
            An = torch.abs(EO)
            sumAn_ThisOrient = sumAn_ThisOrient + An

            if s == 0:
                if abs(noisemethod + 1) < epsilon:
                    tau = float(torch.median(sumAn_ThisOrient).item() / np.sqrt(np.log(4)))
                elif abs(noisemethod + 2) < epsilon:
                    tau = float(_rayleighmode_torch(sumAn_ThisOrient))

            if polarity == 0:
                Energy_ThisOrient = Energy_ThisOrient + torch.abs(torch.real(EO)) - torch.abs(torch.imag(EO))
            elif polarity == 1:
                Energy_ThisOrient = Energy_ThisOrient + torch.real(EO) - torch.abs(torch.imag(EO))
            elif polarity == -1:
                Energy_ThisOrient = Energy_ThisOrient - torch.real(EO) - torch.abs(torch.imag(EO))

        if noisemethod >= 0:
            T = float(noisemethod)
        else:
            totalTau = tau * (1 - (1 / mult)**nscale) / (1 - (1 / mult))
            EstNoiseEnergyMean = totalTau * np.sqrt(np.pi / 2)
            EstNoiseEnergySigma = totalTau * np.sqrt((4 - np.pi) / 2)
            T = float(max(EstNoiseEnergyMean + k * EstNoiseEnergySigma, epsilon))

        Energy_ThisOrient = Energy_ThisOrient - T

        totalSumAn = totalSumAn + sumAn_ThisOrient
        totalEnergy = totalEnergy + Energy_ThisOrient

        if o == 0:
            maxEnergy = Energy_ThisOrient.clone()
        else:
            mask = Energy_ThisOrient > maxEnergy
            orientation[mask] = o
            maxEnergy = torch.maximum(maxEnergy, Energy_ThisOrient)

    phSym = torch.clamp(totalEnergy, min=0.0) / (totalSumAn + epsilon)
    orientation = orientation * np.pi / norient - np.pi / 2

    return _to_numpy(phSym), _to_numpy(orientation), _to_numpy(totalEnergy), T


def _ppdenoise_torch(
    img, nscale=5, norient=6, mult=2.5, minwavelength=2, sigmaonf=0.55,
    dthetaonsigma=1.0, k=3.0, softness=1.0, backend="auto", device=None,
):
    """Torch backend for ppdenoise()."""
    epsilon = 1e-5
    thetaSigma = np.pi / norient / dthetaonsigma

    torch, tdevice, real_dtype, complex_dtype, img_t, IMG = _torch_prepare_image(
        img, backend, device
    )
    rows, cols = img_t.shape

    freq_np, fx_np, fy_np = filtergrids(rows, cols)
    sintheta_np, costheta_np = gridangles(freq_np, fx_np, fy_np)
    freq = torch.as_tensor(freq_np, dtype=real_dtype, device=tdevice)
    sintheta = torch.as_tensor(sintheta_np, dtype=real_dtype, device=tdevice)
    costheta = torch.as_tensor(costheta_np, dtype=real_dtype, device=tdevice)
    eps_freq = torch.finfo(real_dtype).eps

    totalEnergy = torch.zeros((rows, cols), dtype=complex_dtype, device=tdevice)

    RayMean = 0.0
    RayVar = 0.0

    for o in range(norient):
        angl = o * np.pi / norient
        angfilter = _torch_gaussianangularfilter(angl, thetaSigma, sintheta, costheta)

        wavelength = minwavelength
        for s in range(nscale):
            fo = 1.0 / wavelength
            filt = _torch_loggabor(freq, fo, sigmaonf, eps_freq) * angfilter

            EO = torch.fft.ifft2(IMG * filt)
            aEO = torch.abs(EO)

            if s == 0:
                RayMean = float(torch.median(aEO).item() * 0.5 * np.sqrt(-np.pi / np.log(0.5)))
                RayVar = float((4 - np.pi) * (RayMean**2) / np.pi)

            T = (RayMean + k * np.sqrt(RayVar)) / (mult**s)

            mask = aEO > T
            V = softness * T * EO / (aEO + epsilon)
            EO_denoised = EO.clone()
            EO_denoised[mask] = EO_denoised[mask] - V[mask]
            totalEnergy[mask] = totalEnergy[mask] + EO_denoised[mask]

            wavelength *= mult

    return _to_numpy(torch.real(totalEnergy))
