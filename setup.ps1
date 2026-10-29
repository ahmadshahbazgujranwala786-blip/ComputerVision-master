# Portable setup for Smart Attendance System
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
Write-Host '=== Smart Attendance System setup ==='
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python 3.10+ is required. Install Python and enable Add Python to PATH.' }
$py = (Get-Command python).Source
& $py --version
& $py -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e .
if (-not (Test-Path '.env')) { Copy-Item '.env.example' '.env' }
New-Item -ItemType Directory -Force -Path 'data' | Out-Null
Write-Host ''
Write-Host 'Setup complete.'
Write-Host 'Edit .env with this device database configuration, then run run.bat.'
