@echo off
echo Starting Frame Generation Inference on RTX 5070...
echo Using RTX 5070 optimized configuration...
echo Usage: inference_rtx5070.bat input_video.mp4 output_video.mp4 [model_path]
if "%~3"=="" (
    python main.py --mode inference --input %1 --output %2 --config configs/rtx5070_windows.yaml
) else (
    python main.py --mode inference --input %1 --output %2 --model %3 --config configs/rtx5070_windows.yaml
)
pause
