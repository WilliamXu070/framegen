#!/usr/bin/env python3
"""
Setup script for UCF101 dataset preparation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.data.ucf101_dataset import UCF101DatasetManager
from src.data.dataset_validator import DatasetValidator
from src.utils.logger import setup_logger

def main():
    """Setup UCF101 dataset."""
    # Setup logging
    logger = setup_logger()
    logger.info("Setting up UCF101 dataset for frame interpolation training")
    
    # Load configuration
    config_path = "configs/rtx5070_windows.yaml"
    config = Config(config_path)
    
    # Check if UCF101 is enabled
    ucf101_config = config.get('ucf101', {})
    if not ucf101_config.get('enabled', False):
        logger.error("UCF101 dataset is not enabled in configuration")
        logger.info("Please set ucf101.enabled: true in your config file")
        return False
    
    # Initialize UCF101 dataset manager
    logger.info("Initializing UCF101 dataset manager...")
    ucf101_manager = UCF101DatasetManager(config.config)
    
    # Check if dataset already exists
    if ucf101_manager.is_dataset_downloaded():
        logger.info("UCF101 dataset already exists")
        
        # Validate existing dataset
        if ucf101_config.get('quality_check', True):
            logger.info("Validating existing dataset...")
            validator = DatasetValidator(config.config)
            validation_results = validator.validate_ucf101_dataset('data')
            
            if validation_results['overall_valid']:
                logger.info("Existing dataset is valid")
                dataset_info = ucf101_manager.get_dataset_info()
                logger.info("Dataset Info:")
                for key, value in dataset_info.items():
                    logger.info(f"  {key}: {value}")
                return True
            else:
                logger.warning("Existing dataset has validation issues, reprocessing...")
    
    # Prepare dataset
    logger.info("Preparing UCF101 dataset...")
    if not ucf101_manager.prepare_dataset():
        logger.error("Failed to prepare UCF101 dataset")
        return False
    
    # Validate dataset
    if ucf101_config.get('quality_check', True):
        logger.info("Validating prepared dataset...")
        validator = DatasetValidator(config.config)
        validation_results = validator.validate_ucf101_dataset('data')
        
        if not validation_results['overall_valid']:
            logger.warning("Dataset validation found issues:")
            for issue in validation_results['issues']:
                logger.warning(f"  - {issue}")
        
        # Generate validation report
        validator.generate_validation_report(validation_results, 'logs/ucf101_validation_report.json')
        logger.info("Validation report saved to logs/ucf101_validation_report.json")
    
    # Get final dataset info
    dataset_info = ucf101_manager.get_dataset_info()
    logger.info("UCF101 Dataset Setup Complete!")
    logger.info("Dataset Info:")
    for key, value in dataset_info.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("You can now run training with: python examples/train_ucf101_example.py")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
