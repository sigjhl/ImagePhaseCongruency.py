# Julia-to-Python Validation Record

This document records the final Julia-vs-Python numerical validation run performed before removing Julia-related code from the repository.

## Validation Snapshot

- Python/Julia validation commit: `6d29d68`
- Preservation tag: `legacy-julia-transpile-validation`
- Python test run: `95 passed`
- Julia/Python comparison run: `147 total | 145 PASS | 0 FAIL | 2 SKIP`

## Notes

- The two skips were `fillnan_*` reference files due to an upstream Julia runtime issue (`UndefVarError: Images`) in the installed Julia package environment.
- All comparable outputs matched within floating-point tolerance.
- Additional image-level check on `phasecongruency/test_img.png` with `phasesym` showed machine-precision agreement:
  - max abs diff: `3.3306690738754696e-15`
  - thresholded diff (`>1e-10`): `0 / 65024` differing pixels.

## Reproducibility Context

The validation used the repository's Python implementation and a Julia environment with `ImagePhaseCongruency`, `NPZ`, and `FFTW`.
