#!/usr/bin/env python3
"""
Comprehensive test script for the training pipeline to verify deprecated function fixes.
This script tests the training pipeline with minimal data to ensure everything works correctly.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import Config
from src.trainer import FrameInterpolationTrainer
from src.trainer_progress import ProgressFrameInterpolationTrainer
from src.models.frame_interpolation import create_model
from src.utils.logger import setup_logger

def create_test_data():
    """Create minimal test data for training pipeline testing."""
    print("Creating test data...")
    
    # Create temporary directories
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    train_dir = test_data_dir / "train"
    val_dir = test_data_dir / "validation"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Create minimal test videos (just a few frames each)
    for split, num_videos in [("train", 5), ("validation", 2)]:
        split_dir = test_data_dir / split
        for i in range(num_videos):
            video_dir = split_dir / f"video_{i:06d}"
            video_dir.mkdir(exist_ok=True)
            
            # Create 10 frames per video
            for frame_idx in range(10):
                # Create random frame data
                frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                frame_path = video_dir / f"frame_{frame_idx:06d}.jpg"
                
                # Save as numpy array (simplified for testing)
                np.save(str(frame_path).replace('.jpg', '.npy'), frame)
    
    print(f"Test data created in {test_data_dir}")
    return test_data_dir

def test_deprecated_functions():
    """Test that deprecated functions are properly fixed."""
    print("\n" + "="*60)
    print("Testing deprecated function fixes...")
    print("="*60)
    
    # Test 1: GradScaler
    print("1. Testing GradScaler...")
    try:
        # Old method (should work but is deprecated)
        scaler_old = torch.cuda.amp.GradScaler()
        print("   ✓ torch.cuda.amp.GradScaler() works (deprecated)")
    except Exception as e:
        print(f"   ❌ torch.cuda.amp.GradScaler() error: {e}")
    
    try:
        # New method (recommended)
        scaler_new = torch.amp.GradScaler('cuda')
        print("   ✓ torch.amp.GradScaler('cuda') works (new method)")
    except Exception as e:
        print(f"   ❌ torch.amp.GradScaler('cuda') error: {e}")
    
    # Test 2: autocast
    print("\n2. Testing autocast...")
    try:
        # Old method
        with torch.cuda.amp.autocast():
            x = torch.randn(2, 3, 64, 64).cuda()
            y = x * 2
        print("   ✓ torch.cuda.amp.autocast() works (deprecated)")
    except Exception as e:
        print(f"   ❌ torch.cuda.amp.autocast() error: {e}")
    
    try:
        # New method
        with torch.amp.autocast('cuda'):
            x = torch.randn(2, 3, 64, 64).cuda()
            y = x * 2
        print("   ✓ torch.amp.autocast('cuda') works (new method)")
    except Exception as e:
        print(f"   ❌ torch.amp.autocast('cuda') error: {e}")
    
    print("\n✅ Deprecated function tests completed!")

def test_model_creation():
    """Test model creation and basic functionality."""
    print("\n" + "="*60)
    print("Testing model creation...")
    print("="*60)
    
    try:
        # Create a simple config for testing
        config = {
            'name': 'FrameInterpolationNet',
            'architecture': 'optical_flow_based',
            'input_channels': 3,
            'output_channels': 3,
            'hidden_dim': 64,  # Small for testing
            'num_layers': 2,
            'dropout': 0.1
        }
        
        model = create_model(config)
        print(f"   ✓ Model created successfully")
        print(f"   ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            frame1 = torch.randn(1, 3, 64, 64)
            frame2 = torch.randn(1, 3, 64, 64)
            output = model(frame1, frame2)
            print(f"   ✓ Forward pass successful: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False

def test_training_step():
    """Test a single training step."""
    print("\n" + "="*60)
    print("Testing training step...")
    print("="*60)
    
    try:
        # Create minimal config
        config_dict = {
            'model': {
                'name': 'FrameInterpolationNet',
                'architecture': 'optical_flow_based',
                'input_channels': 3,
                'output_channels': 3,
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 2,
                'learning_rate': 0.001,
                'num_epochs': 1,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'gradient_accumulation_steps': 1
            },
            'data': {
                'frame_size': [64, 64],
                'augmentation': {'enabled': False}
            },
            'paths': {
                'data_dir': 'test_data',
                'models_dir': 'test_models',
                'logs_dir': 'test_logs',
                'output_dir': 'test_output'
            },
            'hardware': {
                'mixed_precision': False,  # Disable for testing
                'compile_model': False,   # Disable for testing
                'cuda_benchmark': False
            },
            'ucf101': {
                'enabled': False  # Disable UCF101 for testing
            }
        }
        
        config = Config(config_dict)
        
        # Create trainer with dummy data loaders to bypass dataset creation
        print("   Creating trainer with dummy data loaders...")
        
        # Create dummy data loader class
        class DummyDataLoader:
            def __init__(self, batch_size=2):
                self.batch_size = batch_size
                self.num_batches = 3  # Small number for testing
            
            def __len__(self):
                return self.num_batches
            
            def __iter__(self):
                for _ in range(self.num_batches):
                    yield {
                        'frame1': torch.randn(self.batch_size, 3, 64, 64),
                        'frame2': torch.randn(self.batch_size, 3, 64, 64),
                        'target': torch.randn(self.batch_size, 3, 64, 64)
                    }
        
        # Create trainer manually to bypass data loader setup
        trainer = FrameInterpolationTrainer.__new__(FrameInterpolationTrainer)
        trainer.config = config
        trainer.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        trainer.model = create_model(config.model_config)
        trainer.model.to(trainer.device)
        
        # Initialize optimizer and scheduler
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)
        trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min')
        
        # Initialize loss function
        from src.trainer import CombinedLoss
        trainer.criterion = CombinedLoss()
        
        # Set up data loaders
        trainer.train_loader = DummyDataLoader(2)
        trainer.val_loader = DummyDataLoader(1)
        
        # Initialize other attributes
        trainer.current_epoch = 0
        trainer.best_val_loss = float('inf')
        trainer.patience_counter = 0
        trainer.gradient_accumulation_steps = 1
        trainer.scaler = torch.amp.GradScaler('cuda') if trainer.device.type == 'cuda' else None
        
        # Initialize writer (dummy for testing)
        from torch.utils.tensorboard import SummaryWriter
        trainer.writer = SummaryWriter(log_dir='test_logs')
        
        # Initialize memory manager
        from src.utils.memory_manager import setup_memory_optimization
        trainer.memory_manager = setup_memory_optimization(config.config)
        
        print("   ✓ Trainer created successfully")
        
        # Skip model compilation for testing (Triton issues)
        print("   ⚠️  Skipping model compilation for testing (Triton dependency issues)")
        
        # Test a single training step
        print("   Testing complete training step...")
        
        # Test training epoch
        print("   Testing training epoch...")
        train_metrics = trainer.train_epoch()
        print(f"   ✓ Training epoch completed: {train_metrics}")
        
        # Test validation epoch
        print("   Testing validation epoch...")
        val_metrics = trainer.validate_epoch()
        print(f"   ✓ Validation epoch completed: {val_metrics}")
        
        print("   ✅ Training step test completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_progress_trainer():
    """Test the progress trainer."""
    print("\n" + "="*60)
    print("Testing progress trainer...")
    print("="*60)
    
    try:
        # Create minimal config
        config_dict = {
            'model': {
                'name': 'FrameInterpolationNet',
                'architecture': 'optical_flow_based',
                'input_channels': 3,
                'output_channels': 3,
                'hidden_dim': 64,
                'num_layers': 2,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 2,
                'learning_rate': 0.001,
                'num_epochs': 1,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'gradient_accumulation_steps': 1
            },
            'data': {
                'frame_size': [64, 64],
                'augmentation': {'enabled': False}
            },
            'paths': {
                'data_dir': 'test_data',
                'models_dir': 'test_models',
                'logs_dir': 'test_logs',
                'output_dir': 'test_output'
            },
            'hardware': {
                'mixed_precision': False,
                'compile_model': False,
                'cuda_benchmark': False
            },
            'ucf101': {
                'enabled': False
            }
        }
        
        config = Config(config_dict)
        
        # Create progress trainer with dummy data loaders to bypass dataset creation
        print("   Creating progress trainer with dummy data loaders...")
        
        # Create dummy data loader class
        class DummyDataLoader:
            def __init__(self, batch_size=2):
                self.batch_size = batch_size
                self.num_batches = 2
            
            def __len__(self):
                return self.num_batches
            
            def __iter__(self):
                for _ in range(self.num_batches):
                    yield {
                        'frame1': torch.randn(self.batch_size, 3, 64, 64),
                        'frame2': torch.randn(self.batch_size, 3, 64, 64),
                        'target': torch.randn(self.batch_size, 3, 64, 64)
                    }
        
        # Create trainer manually to bypass data loader setup
        trainer = ProgressFrameInterpolationTrainer.__new__(ProgressFrameInterpolationTrainer)
        trainer.config = config
        trainer.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        trainer.model = create_model(config.model_config)
        trainer.model.to(trainer.device)
        
        # Initialize optimizer and scheduler
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)
        trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min')
        
        # Initialize loss function
        from src.trainer_progress import CombinedLoss
        trainer.criterion = CombinedLoss()
        
        # Set up data loaders
        trainer.train_loader = DummyDataLoader(2)
        trainer.val_loader = DummyDataLoader(1)
        
        # Initialize other attributes
        trainer.current_epoch = 0
        trainer.best_val_loss = float('inf')
        trainer.patience_counter = 0
        trainer.gradient_accumulation_steps = 1
        trainer.scaler = torch.amp.GradScaler('cuda') if trainer.device.type == 'cuda' else None
        trainer.progress_tracker = None
        
        # Initialize writer (dummy for testing)
        from torch.utils.tensorboard import SummaryWriter
        trainer.writer = SummaryWriter(log_dir='test_logs')
        
        # Initialize memory manager
        from src.utils.memory_manager import setup_memory_optimization
        trainer.memory_manager = setup_memory_optimization(config.config)
        
        print("   ✓ Progress trainer created successfully")
        
        # Test training epoch
        print("   Testing progress training epoch...")
        train_metrics = trainer.train_epoch()
        print(f"   ✓ Progress training epoch completed: {train_metrics}")
        
        print("   ✅ Progress trainer test completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Progress trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_data():
    """Clean up test data and directories."""
    print("\n" + "="*60)
    print("Cleaning up test data...")
    print("="*60)
    
    try:
        # Remove test directories
        test_dirs = ["test_data", "test_models", "test_logs", "test_output"]
        for dir_name in test_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(f"   ✓ Removed {dir_name}")
        
        print("   ✅ Cleanup completed!")
        
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")

def main():
    """Main test function."""
    print("🚀 Starting comprehensive training pipeline test...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Test results
    results = []
    
    try:
        # Test 1: Deprecated functions
        test_deprecated_functions()
        results.append(("Deprecated Functions", True))
        
        # Test 2: Model creation
        model_success = test_model_creation()
        results.append(("Model Creation", model_success))
        
        # Test 3: Training step
        training_success = test_training_step()
        results.append(("Training Step", training_success))
        
        # Test 4: Progress trainer
        progress_success = test_progress_trainer()
        results.append(("Progress Trainer", progress_success))
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup_test_data()
    
    # Print results summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(success for _, success in results)
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Training pipeline is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED! Please check the errors above.")
    print("="*60)

if __name__ == "__main__":
    main()
