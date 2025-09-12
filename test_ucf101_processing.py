#!/usr/bin/env python3
"""
Test script for UCF-101 dataset processing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import Config
from src.data.ucf101_processor import UCF101Processor
from src.utils.logger import setup_logger

def test_ucf101_processing():
    """Test UCF-101 dataset processing."""
    logger = setup_logger()
    logger.info("Testing UCF-101 dataset processing...")
    
    # Load configuration
    config = Config('configs/default.yaml')
    
    # Initialize processor
    processor = UCF101Processor(config.config)
    
    # Check if dataset is already processed
    if processor.is_processed():
        logger.info("Dataset already processed, getting info...")
        info = processor.get_processed_info()
        logger.info(f"Processed dataset info: {info}")
        return True
    
    # Check if raw data exists
    if not processor.ucf101_raw_dir.exists():
        logger.error(f"UCF-101 raw directory not found: {processor.ucf101_raw_dir}")
        logger.info("Please ensure the UCF-101 dataset is in the data/UCF-101 directory")
        return False
    
    # Discover videos
    logger.info("Discovering videos...")
    videos = processor.discover_videos()
    logger.info(f"Found {len(videos)} videos")
    
    if len(videos) == 0:
        logger.error("No videos found")
        return False
    
    # Show sample video info
    logger.info("Sample video info:")
    for i, video in enumerate(videos[:3]):
        logger.info(f"  {i+1}. {video['filename']} - {video['class']} - {video['frame_count']} frames - {video['duration']:.2f}s")
    
    # Process dataset
    logger.info("Processing dataset...")
    success = processor.process_dataset()
    
    if success:
        logger.info("Dataset processing completed successfully!")
        info = processor.get_processed_info()
        logger.info(f"Final dataset info: {info}")
        return True
    else:
        logger.error("Dataset processing failed")
        return False

if __name__ == "__main__":
    success = test_ucf101_processing()
    if success:
        print("\n✅ UCF-101 processing test completed successfully!")
    else:
        print("\n❌ UCF-101 processing test failed!")
        sys.exit(1)
