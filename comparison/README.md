# Julia-to-Python Comparison Suite

These scripts enable extensive cross-validation between the original Julia
`ImagePhaseCongruency.jl` package and this Python transpilation.

## How it works

1. **`generate_julia_results.jl`** — Runs every exported function from the Julia
   package with deterministic inputs and saves all outputs as `.npy` files.

2. **`compare_with_julia.py`** — Loads the Julia `.npy` outputs, runs the same
   computations using the Python `phasecongruency` package with identical inputs,
   and reports element-wise differences.

3. **`run_comparison.sh`** — Orchestrates both steps.

## Prerequisites

### Julia side
```bash
# Install Julia from https://julialang.org/downloads/
julia -e 'using Pkg; Pkg.add(["ImagePhaseCongruency", "NPZ"])'
```

### Python side
```bash
pip install numpy scipy
# Install phasecongruency from the parent directory:
pip install -e ..
```

## Usage

```bash
# Full run: generate Julia reference outputs, then compare
./run_comparison.sh

# If you already have julia_results/ from a previous run:
./run_comparison.sh --python-only

# Only generate Julia reference outputs (no Python comparison):
./run_comparison.sh --julia-only

# Or run the Python comparison script directly:
python compare_with_julia.py [path/to/julia_results]
```

## What is compared

The suite covers **every exported function** across all four modules:

| Module | Functions |
|--------|-----------|
| `frequencyfilt` | filtergrids, filtergrid, monogenicfilters, packedmonogenicfilters, lowpassfilter, highpassfilter, bandpassfilter, highboostfilter, loggabor, gridangles, cosineangularfilter, gaussianangularfilter, perfft2, geoseries |
| `syntheticimages` | step2line, circsine, starsine, noiseonf, quantizephase, nophase |
| `utilities` | histtruncate, imgnormalize, replacenan, hysthresh, fillnan |
| `phasecongruency` | phasecongmono, phasesymmono, phasecong3, phasesym, ppdenoise, ppdrc, highpassmonogenic, bandpassmonogenic, monofilt, gaborconvolve |

Functions that use random number generation (noiseonf, nophase) are compared
for shape/type compatibility only, since Julia and Python RNGs differ.

## Tolerances

Comparisons use `numpy.allclose` with `atol=1e-10, rtol=1e-10` for exact
numerical functions, and `atol=1e-8, rtol=1e-8` for iterative/FFT-based
functions where minor floating-point path differences are expected.
