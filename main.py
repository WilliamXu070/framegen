#!/usr/bin/env python3
"""
Frame Generation Application - Main Entry Point
A Python application for generating intermediate frames in low frame rate videos
using advanced computer vision and machine learning techniques.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import Config
from src.trainer import FrameInterpolationTrainer
from src.inference import FrameInterpolationInference
from src.utils.logger import setup_logger

def main():
    """Main entry point for the frame generation application."""
    parser = argparse.ArgumentParser(description='Frame Generation Application')
    parser.add_argument('--mode', choices=['train', 'inference'], required=True,
                       help='Mode: train the model or run inference')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Input video path for inference')
    parser.add_argument('--output', type=str, help='Output video path for inference')
    parser.add_argument('--model', type=str, help='Path to trained model for inference')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger()
    logger.info(f"Starting Frame Generation Application in {args.mode} mode")
    
    # Load configuration
    config = Config(args.config)
    
    if args.mode == 'train':
        trainer = FrameInterpolationTrainer(config)
        trainer.train()
    elif args.mode == 'inference':
        if not args.input or not args.output:
            logger.error("Input and output paths are required for inference mode")
            sys.exit(1)
        
        inference = FrameInterpolationInference(config, args.model)
        inference.process_video(args.input, args.output)

if __name__ == "__main__":
    main()
