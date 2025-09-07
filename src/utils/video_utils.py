"""
Video processing utilities for frame extraction and manipulation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processing utilities for frame extraction and manipulation."""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize video processor.
        
        Args:
            target_size: Target frame size (width, height)
        """
        self.target_size = target_size
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Frame count: {frame_count}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, self.target_size)
            frames.append(frame)
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def extract_frames_generator(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from video file as a generator.
        
        Args:
            video_path: Path to video file
            
        Yields:
            Frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, self.target_size)
            yield frame
        
        cap.release()
    
    def save_frames_as_video(self, frames: List[np.ndarray], output_path: str, fps: float = 30.0):
        """
        Save frames as video file.
        
        Args:
            frames: List of frames
            output_path: Output video path
            fps: Frames per second
        """
        if not frames:
            raise ValueError("No frames to save")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Saved video to: {output_path}")
    
    def create_frame_pairs(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create consecutive frame pairs for training.
        
        Args:
            frames: List of frames
            
        Returns:
            List of consecutive frame pairs
        """
        pairs = []
        for i in range(len(frames) - 1):
            pairs.append((frames[i], frames[i + 1]))
        return pairs
    
    def calculate_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Calculate optical flow between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Optical flow array
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
        return flow
    
    def interpolate_frames_linear(self, frame1: np.ndarray, frame2: np.ndarray, 
                                 num_frames: int) -> List[np.ndarray]:
        """
        Linear interpolation between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            num_frames: Number of frames to interpolate
            
        Returns:
            List of interpolated frames
        """
        frames = []
        for i in range(1, num_frames + 1):
            alpha = i / (num_frames + 1)
            interpolated = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            frames.append(interpolated.astype(np.uint8))
        return frames
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def postprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Postprocess frame from model output.
        
        Args:
            frame: Model output frame
            
        Returns:
            Postprocessed frame
        """
        # Denormalize from [0, 1] to [0, 255]
        frame = (frame * 255.0).astype(np.uint8)
        
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
