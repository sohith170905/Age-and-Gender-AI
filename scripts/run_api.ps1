$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root "venv/Scripts/python.exe"
if (-not (Test-Path $Python)) { $Python = Join-Path $Root ".venv/Scripts/python.exe" }
if (-not (Test-Path $Python)) { throw "Create the project virtual environment and install requirements first." }
if (-not $env:AGE_GENDER_API_KEY) { throw "Set AGE_GENDER_API_KEY in this PowerShell session before starting the optional API." }
& $Python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
