# ============================================================
# setup_rocm_env.ps1
# Rebuild the project .venv with AMD ROCm PyTorch (Windows Edition 7.2)
#
# Prerequisites:
#   1. AMD Adrenalin Edition 26.1.1 driver must be installed first.
#      Download: https://www.amd.com/en/support/download/drivers.html
#   2. Python 3.12 must be in PATH
#
# Usage (from project root in PowerShell as Admin):
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_rocm_env.ps1
#
# NOTE: Total download is ~1.5 GB. This will take several minutes.
# ============================================================

param(
    [string]$VenvPath = ".\.venv"
)

# Direct wheel URLs for Windows + Python 3.12 (cp312) + ROCm 7.2
# The repo.radeon.com/rocm/windows/ path is NOT a PEP 503 index --
# wheels must be installed by direct URL, not via --index-url flag.
$BASE = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2"
$WHEELS = @(
    "$BASE/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl",
    "$BASE/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
    "$BASE/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
    "$BASE/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl"
)

$venvPython = "$VenvPath\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating .venv with Python 3.12..." -ForegroundColor Cyan
    python3.12 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        python -m venv $VenvPath
    }
}

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  AMD ROCm 7.2 -- PyTorch 2.9.1 on Windows Setup" -ForegroundColor Cyan
Write-Host "  ~1.5 GB download -- get a coffee." -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Remove old PyTorch / DirectML wheels (ignore errors if not installed)
Write-Host "[1/4] Removing old PyTorch + DirectML wheels..." -ForegroundColor Yellow
& $venvPython -m pip uninstall torch torchvision torchaudio torch-directml rocm_sdk_core -y 2>&1 | Where-Object { $_ -notmatch "^WARNING" } | Out-Null
Write-Host "      Done." -ForegroundColor Green

# Step 2: Install ROCm SDK + PyTorch wheels by direct URL
# (repo.radeon.com/rocm/windows/ is a raw file listing, not a PEP 503 index)
Write-Host "[2/4] Installing ROCm SDK core + PyTorch 2.9.1 by direct URL..." -ForegroundColor Yellow
Write-Host "      Wheel list:" -ForegroundColor Gray
$WHEELS | ForEach-Object { Write-Host "        $_" -ForegroundColor Gray }
Write-Host ""

& $venvPython -m pip install @WHEELS
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: ROCm/PyTorch installation failed." -ForegroundColor Red
    exit 1
}
Write-Host "      Done." -ForegroundColor Green

# Step 3: Pin numpy for ROCm compatibility
Write-Host "[3/4] Pinning numpy==1.26.4..." -ForegroundColor Yellow
& $venvPython -m pip install "numpy==1.26.4"
Write-Host "      Done." -ForegroundColor Green

# Step 4: Install remaining project requirements (EXCLUDING torch)
Write-Host "[4/4] Installing remaining project requirements (torch excluded)..." -ForegroundColor Yellow
$tempReqs = [System.IO.Path]::GetTempFileName() + ".txt"
Get-Content requirements.txt | Where-Object {
    $_ -notmatch '^\s*(torch|torchvision|torchaudio|rocm)'
} | Set-Content $tempReqs
& $venvPython -m pip install -r $tempReqs
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: requirements install failed." -ForegroundColor Red
    Remove-Item $tempReqs -Force
    exit 1
}
Remove-Item $tempReqs -Force
Write-Host "      Done." -ForegroundColor Green

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  GPU Verification" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

& $venvPython -c @"
import torch, sys
print(f'Python:  {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'ROCm:    {torch.version.hip or "N/A"}')
print()
if torch.cuda.is_available():
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    back = 'ROCm/HIP' if torch.version.hip else 'CUDA'
    print(f'GPU:     {gpu}')
    print(f'VRAM:    {vram:.1f} GB')
    print(f'Backend: {back}')
    x = torch.randn(4, 768, 14, 14, device='cuda')
    print(f'Tensor:  {x.shape} on {x.device}  OK')
    print()
    print('SUCCESS: GPU is ready for training.')
else:
    print('WARNING: torch.cuda.is_available() = False')
    print('  Check: AMD Adrenalin 26.1.1 driver installed and rebooted.')
"@

Write-Host ""
Write-Host "Next step: python src/train/train.py --config config.yaml --task detect" -ForegroundColor Cyan
