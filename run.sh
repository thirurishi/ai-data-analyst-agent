#!/usr/bin/env bash
# macOS/Linux quick start: creates the virtual environment on first run, then starts the app.
set -euo pipefail
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -q -r requirements.txt
echo "Open http://127.0.0.1:8000 in your browser"
.venv/bin/python -m uvicorn backend.app.main:app --reload
