#!/usr/bin/env python3
"""
Windows RTX 5070 setup script for Frame Generation Application.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_cuda_installation():
    """Check if CUDA is properly installed."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available!")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("❌ CUDA is not available")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False

def install_requirements():
    """Install requirements with RTX 5070 optimizations."""
    print("Installing requirements for RTX 5070...")
    
    # Install PyTorch with CUDA support
    pytorch_cmd = [
        "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    try:
        subprocess.run(pytorch_cmd, check=True)
        print("✅ PyTorch with CUDA installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False
    
    # Install other requirements
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Other requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    
    return True

def setup_directories():
    """Create necessary directories."""
    directories = [
        "data/train",
        "data/validation", 
        "data/test",
        "models",
        "logs",
        "output",
        "examples",
        "scripts",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create placeholder files
    placeholder_files = [
        "data/train/.gitkeep",
        "data/validation/.gitkeep",
        "data/test/.gitkeep",
        "models/.gitkeep",
        "logs/.gitkeep",
        "output/.gitkeep"
    ]
    
    for file_path in placeholder_files:
        Path(file_path).touch()
        print(f"✅ Created placeholder file: {file_path}")

def create_windows_batch_files():
    """Create Windows batch files for easy execution."""
    
    # Training batch file
    train_batch = """@echo off
echo Starting Frame Generation Training on RTX 5070...
python main.py --mode train --config configs/rtx5070_windows.yaml
pause
"""
    
    with open("train_rtx5070.bat", "w") as f:
        f.write(train_batch)
    print("✅ Created train_rtx5070.bat")
    
    # Inference batch file
    inference_batch = """@echo off
echo Starting Frame Generation Inference on RTX 5070...
echo Usage: inference_rtx5070.bat input_video.mp4 output_video.mp4 [model_path]
if "%~3"=="" (
    python main.py --mode inference --input %1 --output %2 --config configs/rtx5070_windows.yaml
) else (
    python main.py --mode inference --input %1 --output %2 --model %3 --config configs/rtx5070_windows.yaml
)
pause
"""
    
    with open("inference_rtx5070.bat", "w") as f:
        f.write(inference_batch)
    print("✅ Created inference_rtx5070.bat")

def optimize_windows_performance():
    """Apply Windows-specific performance optimizations."""
    print("Applying Windows performance optimizations...")
    
    # Set environment variables for better performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
    
    print("✅ Windows performance optimizations applied")

def main():
    """Main setup function."""
    print("🚀 Setting up Frame Generation Application for Windows RTX 5070")
    print("=" * 60)
    
    # Check CUDA
    if not check_cuda_installation():
        print("\n⚠️  CUDA not detected. Installing PyTorch with CUDA support...")
        if not install_requirements():
            print("❌ Setup failed. Please install CUDA manually.")
            return False
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        print("❌ Failed to install requirements")
        return False
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Create batch files
    print("\n🔧 Creating Windows batch files...")
    create_windows_batch_files()
    
    # Apply optimizations
    print("\n⚡ Applying performance optimizations...")
    optimize_windows_performance()
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your training videos in data/train/")
    print("2. Run train_rtx5070.bat to start training")
    print("3. Use inference_rtx5070.bat for video processing")
    print("\nFor more information, see README.md")
    
    return True

if __name__ == "__main__":
    main()
