# Agent Instructions: Python-Only Maintenance

## Context

This repository maintains a Python package (`phasecongruency`) for phase-congruency-based image processing.

## Expectations

1. Prioritize Python package correctness and tests in `tests/`.
2. Use `uv` for environment and test execution.
3. Keep public API behavior stable unless explicitly requested.
4. Preserve configurable parameter surfaces for core algorithms (`phasecongmono`, `phasecong3`, `phasesymmono`, `phasesym`, `ppdrc`, `ppdenoise`, `monofilt`, `gaborconvolve`, `highpassmonogenic`, `bandpassmonogenic`).
5. Do not hardcode algorithm constants that are already exposed as function parameters.
6. If any API/signature/default changes are made, update `README.md` examples and add/adjust tests in `tests/`.

## Canonical Checks

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests -q
```
