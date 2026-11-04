# Windows quick start: creates the virtual environment on first run, then starts the app.
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python -m pip install -q -r requirements.txt
Write-Host "Open http://127.0.0.1:8000 in your browser"
.\.venv\Scripts\python -m uvicorn backend.app.main:app --reload
