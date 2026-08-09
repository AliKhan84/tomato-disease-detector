#!/usr/bin/env bash
# Launch the Streamlit app with the interpreter that actually has TensorFlow.
#
# Why this script exists: `conda activate tf-uetm` sets CONDA_PREFIX and changes the
# prompt, but pyenv's shims are prepended to PATH in ~/.bashrc, so a bare `streamlit`
# still resolves to pyenv's Python 3.14 -- which has no TensorFlow and fails with
# `ModuleNotFoundError: No module named 'tensorflow'`. `conda run -n tf-uetm` inherits
# the same PATH and is shadowed too. Calling the env's python by absolute path is the
# only launch that cannot be hijacked.
#
# It also cds to the repo root, because Streamlit resolves .streamlit/config.toml
# against the working directory -- launching from src/ silently drops the theme.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${TOMATO_PYTHON:-$HOME/anaconda3/envs/tf-uetm/bin/python}"

if [ ! -x "$PY" ]; then
    echo "error: no interpreter at $PY" >&2
    echo "       Set TOMATO_PYTHON to a Python 3.11-3.13 that has the requirements:" >&2
    echo "         TOMATO_PYTHON=/path/to/python ./run_app.sh" >&2
    exit 1
fi

if ! "$PY" -c "import tensorflow, streamlit" 2>/dev/null; then
    echo "error: $PY is missing tensorflow or streamlit" >&2
    echo "       Install them into that env:  $PY -m pip install -r requirements.txt" >&2
    exit 1
fi

cd "$ROOT"
exec "$PY" -m streamlit run src/streamlit_app.py "$@"
