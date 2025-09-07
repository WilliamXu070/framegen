@echo off
echo Starting UCF101 Frame Generation Training on RTX 5070...
echo Using UCF101 Action Recognition Dataset...
echo.

REM Check if dataset is prepared
if not exist "data\train\video_000000" (
    echo UCF101 dataset not found. Setting up dataset first...
    python scripts\setup_ucf101.py
    if errorlevel 1 (
        echo Dataset setup failed. Please check the logs.
        pause
        exit /b 1
    )
    echo.
)

echo Starting training with UCF101 dataset...
python examples\train_ucf101_example.py
pause
