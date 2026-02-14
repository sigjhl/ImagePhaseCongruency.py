#!/bin/bash
#
# Run the Julia-to-Python comparison suite.
#
# This script:
#   1. Runs the Julia generator to produce reference .npy outputs
#   2. Runs the Python comparator to diff against the Python package
#
# Usage:
#   ./run_comparison.sh                # Full run (Julia + Python)
#   ./run_comparison.sh --python-only  # Skip Julia, use existing results
#   ./run_comparison.sh --julia-only   # Only generate Julia results
#
# Prerequisites:
#   - Julia with ImagePhaseCongruency and NPZ packages installed
#   - Python with numpy, scipy, and the phasecongruency package
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/julia_results"

RUN_JULIA=true
RUN_PYTHON=true

for arg in "$@"; do
    case "$arg" in
        --python-only) RUN_JULIA=false ;;
        --julia-only)  RUN_PYTHON=false ;;
        --help|-h)
            echo "Usage: $0 [--python-only|--julia-only|--help]"
            echo ""
            echo "  --python-only  Skip Julia generation, use existing .npy files"
            echo "  --julia-only   Only run Julia to generate reference outputs"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Julia-to-Python Comparison Suite"
echo "========================================"
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Step 1: Generate Julia reference outputs
if [ "$RUN_JULIA" = true ]; then
    echo "Step 1: Generating Julia reference outputs..."
    echo "----------------------------------------"

    if ! command -v julia &>/dev/null; then
        echo "ERROR: julia not found in PATH"
        echo "Install Julia from https://julialang.org/downloads/"
        echo ""
        echo "Or run with --python-only if you already have julia_results/"
        exit 1
    fi

    # Check that required Julia packages are available
    julia -e 'using ImagePhaseCongruency; using NPZ' 2>/dev/null || {
        echo "ERROR: Required Julia packages not found."
        echo "Install them with:"
        echo '  julia -e '\''using Pkg; Pkg.add(["ImagePhaseCongruency", "NPZ"])'\'''
        exit 1
    }

    julia "${SCRIPT_DIR}/generate_julia_results.jl" "${RESULTS_DIR}"
    echo ""
else
    echo "Step 1: Skipped (--python-only)"
    if [ ! -d "${RESULTS_DIR}" ]; then
        echo "ERROR: ${RESULTS_DIR} does not exist. Run without --python-only first."
        exit 1
    fi
    echo ""
fi

# Step 2: Run Python comparison
if [ "$RUN_PYTHON" = true ]; then
    echo "Step 2: Comparing Python outputs against Julia..."
    echo "----------------------------------------"

    if ! python3 -c "import phasecongruency" 2>/dev/null; then
        echo "WARNING: phasecongruency package not importable. Trying with PYTHONPATH..."
        export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"
    fi

    python3 "${SCRIPT_DIR}/compare_with_julia.py" "${RESULTS_DIR}"
else
    echo "Step 2: Skipped (--julia-only)"
fi

echo ""
echo "Done!"
