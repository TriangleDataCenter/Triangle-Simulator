#!/usr/bin/env bash
# Run from any working directory. An activated environment takes precedence.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -n "${TRIANGLE_PYTHON:-}" ]]; then
    PYTHON_BIN="$TRIANGLE_PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -c 'import tkinter, numpy, scipy, matplotlib' || {
    echo 'GUI dependencies are missing. Activate the project Python environment (with tkinter) first.' >&2
    exit 1
}
exec "$PYTHON_BIN" "$SCRIPT_DIR/gui_app.py" "$@"
