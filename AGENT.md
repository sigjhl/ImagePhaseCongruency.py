# Agent Instructions: Julia-to-Python Fidelity Validation

## Context

This repo is a Python transpilation of [ImagePhaseCongruency.jl](https://github.com/peterkovesi/ImagePhaseCongruency.jl), a Julia package for phase congruency-based image feature detection. The transpilation is complete and has 88 unit tests + 106 cross-validation checks passing, but **has not yet been validated against the actual Julia package**.

## What Has Been Done

1. **Full transpilation** — All 4 Julia source modules → `phasecongruency/` Python package
2. **Unit tests** — `tests/` with 88 passing tests
3. **Cross-validation tests** — `tests/test_cross_validation.py` with 106 checks using independent reference implementations
4. **Bug audit** — Found and fixed 4 transpilation bugs (atan vs arctan2, array ordering, connectivity)
5. **Comparison scripts** — `comparison/` directory with Julia + Python scripts ready to run

## What You Need To Do

### Primary Task: Run the Julia-vs-Python comparison

The comparison scripts exist but have never been executed (the previous environment had no network access to install Julia). You need to:

1. **Install Julia** (if not already available):
   ```bash
   # Download from https://julialang.org/downloads/
   # Or use juliaup: curl -fsSL https://install.julialang.org | sh
   ```

2. **Install Julia dependencies**:
   ```bash
   julia -e 'using Pkg; Pkg.add(["ImagePhaseCongruency", "NPZ", "FFTW"])'
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -e .  # Install phasecongruency package
   pip install numpy scipy pytest
   ```

4. **Run the comparison**:
   ```bash
   cd comparison
   ./run_comparison.sh
   ```
   Or step-by-step:
   ```bash
   julia comparison/generate_julia_results.jl comparison/julia_results
   python comparison/compare_with_julia.py comparison/julia_results
   ```

5. **Analyze results** — The Python script reports per-function max absolute differences. Any FAIL needs investigation.

### If Failures Are Found

Investigate each failure by:
- Checking if it's a known transpilation pitfall (see below)
- Reading the corresponding Julia source in `src/` and Python source in `phasecongruency/`
- Fix the Python code, add a regression test in `tests/test_cross_validation.py`, and re-run

### Secondary Task: Run existing tests

```bash
pytest tests/ -v
```

All 88 tests should pass. If any fail, fix them before proceeding.

## Known Transpilation Pitfalls

These were already fixed, but watch for similar patterns:

| Issue | Julia | Python (correct) | Python (wrong) |
|-------|-------|-------------------|----------------|
| Single-arg atan | `atan(y/x)` → (-π/2, π/2) | `np.arctan(y/x)` | `np.arctan2(y, x)` |
| Two-arg atan | `atan(y, x)` → (-π, π) | `np.arctan2(y, x)` | `np.arctan(y/x)` |
| Comprehension order | `[f(x,y) for x=xr, y=yr]` | x=rows, y=cols | meshgrid swaps them |
| Label connectivity | `strel_box((3,3))` = 8-conn | `structure=np.ones((3,3))` | default = 4-conn |

## File Layout

```
phasecongruency/          # Python package (4 modules)
  __init__.py             # Exports all ~30 functions
  frequencyfilt.py        # Frequency domain filters
  syntheticimages.py      # Synthetic test image generators
  phasecongruency.py      # Core phase congruency algorithms
  utilities.py            # Utility functions
tests/                    # pytest test suite
  test_frequencyfilt.py   # 23 tests
  test_syntheticimages.py # 14 tests
  test_phasecongruency.py # 16 tests
  test_utilities.py       # 10 tests
  test_cross_validation.py # 106 cross-validation checks
comparison/               # Julia-vs-Python comparison suite
  generate_julia_results.jl  # Saves Julia outputs as .npy files
  compare_with_julia.py      # Loads .npy files, diffs against Python
  run_comparison.sh          # Orchestrates both scripts
src/                      # Original Julia source (reference)
pyproject.toml            # Python package config
```

## Branch

All work is on: `claude/julia-to-python-transpile-okxcJ`
