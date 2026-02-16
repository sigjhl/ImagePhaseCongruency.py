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
```

## Parameter Configurability

Yes. Core algorithm parameters are configurable in Python, matching the Julia package intent.

- Use keyword arguments on: `phasecongmono`, `phasecong3`, `phasesymmono`, `phasesym`, `ppdrc`, `ppdenoise`, `monofilt`, `gaborconvolve`, `highpassmonogenic`, and `bandpassmonogenic`.
- Default values follow the Julia defaults used in the transpilation.
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
