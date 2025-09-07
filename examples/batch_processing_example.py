#!/usr/bin/env python3
"""
Example script for batch processing multiple videos.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.inference import FrameInterpolationInference
from src.utils.logger import setup_logger

def main():
    """Example batch processing script."""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting batch processing example")
    
    # Configuration
    config_path = "configs/default.yaml"
    model_path = "models/best_model.pth"
    input_dir = "data/test"
    output_dir = "output/batch_processed"
    
    # Check if files exist
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please train a model first or provide a valid model path")
        return
    
    if not Path(input_dir).exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.info("Please provide a valid input directory")
        return
    
    # Load configuration
    config = Config(config_path)
    
    # Initialize inference
    inference = FrameInterpolationInference(config, model_path)
    
    # Process videos in batch
    try:
        logger.info(f"Processing videos in directory: {input_dir}")
        inference.batch_process_videos(input_dir, output_dir)
        logger.info(f"Batch processing completed! Output saved to: {output_dir}")
    except Exception as e:
        logger.error(f"Batch processing failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
