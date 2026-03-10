# E-Netra V4 — Dashboard Watcher Launcher
# Starts the watcher which rebuilds every 5 min and serves on port 8765
Set-Location $PSScriptRoot
$env:HOME = $env:USERPROFILE

Write-Host "Starting E-Netra Dashboard Watcher..." -ForegroundColor Cyan
python -X utf8 scripts/watch_rebuild.py --interval 300 --port 8765
