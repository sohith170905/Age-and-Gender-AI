$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root "venv/Scripts/python.exe"
if (-not (Test-Path $Python)) { $Python = Join-Path $Root ".venv/Scripts/python.exe" }
if (-not (Test-Path $Python)) { throw "Create the project virtual environment and install requirements first." }
& $Python -m streamlit run (Join-Path $Root "app.py") --server.address 127.0.0.1 --server.port 8501
