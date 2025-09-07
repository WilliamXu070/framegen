#!/usr/bin/env python3
"""
Example script for training a frame interpolation model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.trainer import FrameInterpolationTrainer
from src.utils.logger import setup_logger

def main():
    """Example training script."""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting training example")
    
    # Load configuration
    config_path = "configs/default.yaml"
    config = Config(config_path)
    
    # Print configuration summary
    logger.info("Configuration loaded:")
    logger.info(f"  Model: {config.get('model.name', 'Unknown')}")
    logger.info(f"  Architecture: {config.get('model.architecture', 'Unknown')}")
    logger.info(f"  Batch size: {config.get('training.batch_size', 'Unknown')}")
    logger.info(f"  Learning rate: {config.get('training.learning_rate', 'Unknown')}")
    logger.info(f"  Epochs: {config.get('training.num_epochs', 'Unknown')}")
    
    # Initialize trainer
    trainer = FrameInterpolationTrainer(config)
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
