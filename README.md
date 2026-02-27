# phasecongruency

Python package for phase-congruency-based feature detection and phase-preserving image enhancement.

## Installation

From source with pip:

```bash
pip install .
```

Editable install for development:

```bash
pip install -e ".[dev]"
```

Or with uv:

```bash
uv sync --extra dev
```

GPU backend (PyTorch / Apple Metal MPS):

```bash
pip install "phasecongruency[gpu]"
```

## Quick Start

```python
from phasecongruency import phasecongmono, phasecong3, phasesym, step2line

img = step2line(256)

# Fast monogenic phase congruency (defaults)
PC, orientation, phase_type, T = phasecongmono(img)

# Tuned monogenic phase congruency (all key parameters configurable)
PC2, orientation2, phase_type2, T2 = phasecongmono(
    img,
    nscale=5,
    minwavelength=3,
    mult=2.1,
    sigmaonf=0.55,
    k=3.0,
    noisemethod=-1,
    cutoff=0.5,
    g=10.0,
    deviationgain=1.5,
)

# Oriented phase congruency (edge/corner moments)
M, m, or_, feat_type, EO, T3 = phasecong3(
    img,
    nscale=4,
    norient=6,
    minwavelength=3,
    mult=2.1,
    sigmaonf=0.55,
    k=2.0,
    cutoff=0.5,
    g=10.0,
    noisemethod=-1,
)

# Phase symmetry (blob/line-like structures)
phSym, orient, totalEnergy, Tps = phasesym(img)

# Optional GPU backend (auto-selects MPS on Apple Silicon if available)
PC_gpu, orientation_gpu, phase_type_gpu, T_gpu = phasecongmono(
    img,
    backend="auto",
)

# Force Apple Metal backend explicitly
M_gpu, m_gpu, or_gpu, feat_gpu, EO_gpu, T3_gpu = phasecong3(
    img,
    backend="torch-mps",
)

# Low-memory mode when EO outputs are not needed
M_lm, m_lm, or_lm, feat_lm, EO_lm, T_lm = phasecong3(
    img,
    return_eo=False,
)
assert EO_lm is None
```

`backend="auto"` falls back to NumPy if PyTorch is not installed.

When PyTorch is available, `backend="auto"` benchmarks NumPy vs torch once
per function + image shape + parameter set, then caches the winner on disk for
reuse across runs.

- Default cache path:
  - macOS/Linux: `~/.cache/phasecongruency/auto_backend_cache.json`
  - Windows: `%LOCALAPPDATA%\\phasecongruency\\auto_backend_cache.json`
- Override cache path with env var:
  - `PHASECONGRUENCY_AUTO_CACHE_PATH=/path/to/cache.json`

## Parameter Configurability

Core algorithm parameters are exposed as keyword arguments in Python.

- Use keyword arguments on: `phasecongmono`, `phasecong3`, `phasesymmono`, `phasesym`, `ppdrc`, `ppdenoise`, `monofilt`, `gaborconvolve`, `highpassmonogenic`, and `bandpassmonogenic`.
- Default values are defined in each function signature.
- GPU-capable functions also accept `backend` (`"numpy"`, `"torch"`, `"torch-mps"`, `"auto"`) and optional `device`.
- Inspect any callable signature directly:

```python
from inspect import signature
from phasecongruency import phasecongmono
print(signature(phasecongmono))
```

## Run Tests

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests -q
# or
pytest tests -q
```

## Scope

This repository now maintains the Python implementation only. The historical Julia transpilation/validation workflow has been retired from the active tree.

## Validation Provenance

A Julia-vs-Python fidelity validation was completed before this cleanup. See:

- `docs/validation.md`

## Attribution

This work is based on Peter Kovesi's phase congruency methods and was originally transpiled from:

- [ImagePhaseCongruency.jl](https://github.com/peterkovesi/ImagePhaseCongruency.jl)
