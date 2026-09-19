#!/usr/bin/env bash
# Convenience wrapper: sets up the venv on first run, then launches the sim.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "creating .venv (Python 3.12; Taichi does not yet support 3.13+)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    python3.12 -m venv .venv
    .venv/bin/pip install -r requirements.txt
  fi
fi

exec .venv/bin/python main.py "$@"
