@echo off
REM Migration script for RTX 5070 compatibility
REM This script helps upgrade PyTorch and dependencies for RTX 5070 (Blackwell sm_120)

echo ========================================
echo RTX 5070 Dependency Migration Script
echo ========================================
echo.

echo Current PyTorch configuration:
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
echo.

echo WARNING: This will uninstall current PyTorch and install new versions
echo Press Ctrl+C to cancel, or
pause

echo.
echo [Step 1/5] Creating backup of current packages...
pip freeze > old_requirements_backup.txt
echo Backup saved to old_requirements_backup.txt
echo.

echo [Step 2/5] Uninstalling old PyTorch...
pip uninstall torch torchvision torchaudio -y
echo.

echo [Step 3/5] Installing PyTorch 2.4+ with CUDA 12.6 for RTX 5070...
pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 --index-url https://download.pytorch.org/whl/cu126
echo.

echo [Step 4/5] Upgrading PyTorch Lightning to 2.x...
pip install pytorch-lightning>=2.1.0 torchmetrics>=1.0.0 lightning-utilities>=0.10.0 --upgrade
echo.

echo [Step 5/5] Updating other dependencies...
if exist requirements_rtx5070.txt (
    pip install -r requirements_rtx5070.txt
) else (
    echo requirements_rtx5070.txt not found, skipping...
)
echo.

echo ========================================
echo Migration complete! Verifying installation...
echo ========================================
echo.

python -c "import torch; import pytorch_lightning as pl; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'Lightning: {pl.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

echo ========================================
echo IMPORTANT: Check for warnings above
echo If you see "sm_120 is not compatible", you may need PyTorch nightly:
echo   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
echo.
echo See RTX5070_MIGRATION_GUIDE.md for detailed migration steps
echo ========================================
pause
