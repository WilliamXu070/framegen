#!/usr/bin/env python3
"""
Example script for running inference on a video.
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
    """Example inference script."""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting inference example")
    
    # Configuration
    config_path = "configs/default.yaml"
    model_path = "models/best_model.pth"
    input_video = "data/test/input_video.mp4"
    output_video = "output/interpolated_video.mp4"
    
    # Check if files exist
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please train a model first or provide a valid model path")
        return
    
    if not Path(input_video).exists():
        logger.error(f"Input video not found: {input_video}")
        logger.info("Please provide a valid input video path")
        return
    
    # Load configuration
    config = Config(config_path)
    
    # Print configuration summary
    logger.info("Configuration loaded:")
    logger.info(f"  Interpolation factor: {config.get('inference.interpolation_factor', 'Unknown')}")
    logger.info(f"  Temporal smoothing: {config.get('inference.temporal_smoothing', 'Unknown')}")
    logger.info(f"  Post-processing: {config.get('inference.post_processing', 'Unknown')}")
    
    # Initialize inference
    inference = FrameInterpolationInference(config, model_path)
    
    # Process video
    try:
        logger.info(f"Processing video: {input_video}")
        inference.process_video(input_video, output_video)
        logger.info(f"Inference completed! Output saved to: {output_video}")
    except Exception as e:
        logger.error(f"Inference failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
