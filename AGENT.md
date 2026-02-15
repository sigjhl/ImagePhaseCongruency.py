# Agent Instructions: Python-Only Maintenance

## Context

This repository maintains a Python package (`phasecongruency`) for phase-congruency-based image processing.

## Expectations

1. Prioritize Python package correctness and tests in `tests/`.
2. Use `uv` for environment and test execution.
3. Keep public API behavior stable unless explicitly requested.

## Canonical Checks

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests -q
```
