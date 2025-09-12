#!/usr/bin/env python3
"""
Test script for light loading mode.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import Config
from src.data.ucf101_dataset_loader import create_ucf101_data_loaders
from src.utils.logger import setup_logger

def test_light_loading():
    """Test light loading mode."""
    logger = setup_logger()
    logger.info("Testing light loading mode...")
    
    # Load configuration
    config = Config('configs/default.yaml')
    
    # Test normal loading
    logger.info("Testing normal loading...")
    train_loader, val_loader = create_ucf101_data_loaders(config.config, light_loading=False)
    logger.info(f"Normal loading - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Test light loading
    logger.info("Testing light loading...")
    train_loader_light, val_loader_light = create_ucf101_data_loaders(config.config, light_loading=True)
    logger.info(f"Light loading - Train: {len(train_loader_light.dataset)}, Val: {len(val_loader_light.dataset)}")
    
    # Verify reduction
    train_reduction = len(train_loader.dataset) - len(train_loader_light.dataset)
    val_reduction = len(val_loader.dataset) - len(val_loader_light.dataset)
    
    logger.info(f"Training dataset reduced by: {train_reduction} samples")
    logger.info(f"Validation dataset reduced by: {val_reduction} samples")
    
    if len(train_loader_light.dataset) <= 800:  # 100 videos * 8 samples
        logger.info("Light loading working correctly!")
        return True
    else:
        logger.error("Light loading not working - dataset too large")
        return False

if __name__ == "__main__":
    success = test_light_loading()
    if success:
        print("\n[SUCCESS] Light loading test completed successfully!")
    else:
        print("\n[FAILED] Light loading test failed!")
        sys.exit(1)
