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
from phasecongruency import phasecongmono, phasesym, step2line

img = step2line(256)
PC, orientation, phase_type, T = phasecongmono(img)
phSym, orient, totalEnergy, Tps = phasesym(img)
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
