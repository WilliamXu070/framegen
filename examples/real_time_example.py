#!/usr/bin/env python3
"""
Example script for real-time frame interpolation.
"""

import sys
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.inference import RealTimeFrameInterpolation
from src.utils.logger import setup_logger

def main():
    """Example real-time processing script."""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting real-time processing example")
    
    # Configuration
    config_path = "configs/default.yaml"
    model_path = "models/best_model.pth"
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please train a model first or provide a valid model path")
        return
    
    # Load configuration
    config = Config(config_path)
    
    # Initialize real-time processor
    rt_processor = RealTimeFrameInterpolation(config, model_path)
    
    # Open video capture (use 0 for webcam, or provide video file path)
    video_source = 0  # Change to video file path if needed
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        logger.error("Could not open video source")
        return
    
    logger.info("Real-time processing started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break
            
            # Process frame
            processed_frame = rt_processor.process_frame(frame)
            
            # Display frame
            cv2.imshow('Real-time Frame Interpolation', processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Real-time processing interrupted by user")
    except Exception as e:
        logger.error(f"Real-time processing failed with error: {e}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Real-time processing ended")

if __name__ == "__main__":
    main()
