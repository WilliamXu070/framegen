#!/usr/bin/env python3
"""
Example script for training with UCF101 dataset.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.trainer import FrameInterpolationTrainer
from src.data.ucf101_dataset import UCF101DatasetManager
from src.data.dataset_validator import DatasetValidator
from src.utils.logger import setup_logger

def main():
    """Example UCF101 training script."""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting UCF101 training example")
    
    # Load configuration
    config_path = "configs/rtx5070_windows.yaml"  # Use RTX 5070 config
    config = Config(config_path)
    
    # Check if UCF101 is enabled
    ucf101_config = config.get('ucf101', {})
    if not ucf101_config.get('enabled', False):
        logger.error("UCF101 dataset is not enabled in configuration")
        logger.info("Please set ucf101.enabled: true in your config file")
        return
    
    # Print configuration summary
    logger.info("UCF101 Configuration:")
    logger.info(f"  Enabled: {ucf101_config.get('enabled', False)}")
    logger.info(f"  Interpolation Factor: {ucf101_config.get('interpolation_factor', 4)}")
    logger.info(f"  Min Frames per Video: {ucf101_config.get('min_frames_per_video', 10)}")
    logger.info(f"  Train Ratio: {ucf101_config.get('train_ratio', 0.7)}")
    logger.info(f"  Val Ratio: {ucf101_config.get('val_ratio', 0.15)}")
    logger.info(f"  Test Ratio: {ucf101_config.get('test_ratio', 0.15)}")
    
    # Initialize UCF101 dataset manager
    logger.info("Initializing UCF101 dataset manager...")
    ucf101_manager = UCF101DatasetManager(config.config)
    
    # Prepare dataset
    logger.info("Preparing UCF101 dataset...")
    if not ucf101_manager.prepare_dataset():
        logger.error("Failed to prepare UCF101 dataset")
        return
    
    # Validate dataset
    if ucf101_config.get('quality_check', True):
        logger.info("Validating UCF101 dataset...")
        validator = DatasetValidator(config.config)
        validation_results = validator.validate_ucf101_dataset('data')
        
        if not validation_results['overall_valid']:
            logger.warning("Dataset validation found issues:")
            for issue in validation_results['issues']:
                logger.warning(f"  - {issue}")
        
        # Generate validation report
        validator.generate_validation_report(validation_results, 'logs/ucf101_validation_report.json')
        logger.info("Validation report saved to logs/ucf101_validation_report.json")
    
    # Get dataset info
    dataset_info = ucf101_manager.get_dataset_info()
    logger.info("UCF101 Dataset Info:")
    for key, value in dataset_info.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = FrameInterpolationTrainer(config)
    
    # Start training
    try:
        logger.info("Starting UCF101 training...")
        trainer.train()
        logger.info("UCF101 training completed successfully!")
    except KeyboardInterrupt:
        logger.info("UCF101 training interrupted by user")
    except Exception as e:
        logger.error(f"UCF101 training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
