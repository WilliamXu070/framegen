# PowerShell script to recompile CuPy for RTX 5070 (compute capability 12.0)
# Run this script in PowerShell

Write-Host "=== Recompiling CuPy for RTX 5070 (sm_120) ===" -ForegroundColor Cyan

# Step 1: Check current setup
Write-Host "`n[1/5] Checking current setup..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'Compute Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"N/A\"}')"

# Step 2: Check CUDA Toolkit
Write-Host "`n[2/5] Checking CUDA Toolkit..." -ForegroundColor Yellow
nvcc --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: nvcc not found. Make sure CUDA Toolkit is installed." -ForegroundColor Red
}

# Step 3: Uninstall existing CuPy
Write-Host "`n[3/5] Uninstalling existing CuPy..." -ForegroundColor Yellow
pip uninstall cupy -y

# Step 4: Set environment variable for RTX 5070 (compute capability 12.0)
Write-Host "`n[4/5] Setting environment variable for compute capability 12.0..." -ForegroundColor Yellow
$env:CUPY_NVCC_GENERATE_CODE = "arch=compute_120,code=sm_120"
Write-Host "CUPY_NVCC_GENERATE_CODE = $env:CUPY_NVCC_GENERATE_CODE" -ForegroundColor Green

# Step 5: Install CuPy from source
Write-Host "`n[5/5] Installing CuPy from source (this may take 10-30 minutes)..." -ForegroundColor Yellow
Write-Host "This will compile CuPy with support for your RTX 5070 GPU." -ForegroundColor Cyan
pip install cupy --no-binary cupy

# Step 6: Verify installation
Write-Host "`n=== Verification ===" -ForegroundColor Cyan
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); print(f'CUDA version: {cp.cuda.runtime.runtimeGetVersion()}'); print(f'CuPy CUDA available: {cp.cuda.is_available()}')"

Write-Host "`n=== Done! ===" -ForegroundColor Green
Write-Host "If all checks passed, CuPy should now work with your RTX 5070." -ForegroundColor Green
