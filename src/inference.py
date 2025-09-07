"""
Inference pipeline for frame interpolation and video generation.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import logging
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm

from src.models.frame_interpolation import create_model
from src.utils.video_utils import VideoProcessor
from src.config import Config

logger = logging.getLogger(__name__)

class FrameInterpolationInference:
    """Inference class for frame interpolation."""
    
    def __init__(self, config: Config, model_path: Optional[str] = None):
        """
        Initialize inference pipeline.
        
        Args:
            config: Configuration object
            model_path: Path to trained model checkpoint
        """
        self.config = config
        self.device = self._setup_device()
        
        # Initialize model
        self.model = create_model(config.model_config)
        self.model.to(self.device)
        
        # Load model weights
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("No model path provided. Using untrained model.")
        
        # RTX 5070 model compilation for better inference performance
        if config.get('hardware.compile_model', False) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                logger.info("Model compiled for RTX 5070 inference optimization")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        self.model.eval()
        
        # Initialize video processor
        frame_size = config.data_config.get('frame_size', [256, 256])
        self.video_processor = VideoProcessor(tuple(frame_size))
        
        # Inference configuration
        self.inference_config = config.get('inference', {})
        self.interpolation_factor = self.inference_config.get('interpolation_factor', 4)
        self.temporal_smoothing = self.inference_config.get('temporal_smoothing', True)
        self.post_processing = self.inference_config.get('post_processing', True)
        
        logger.info(f"Inference pipeline initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup inference device with RTX 5070 optimizations."""
        training_config = self.config.training_config
        device_str = training_config.get('device', 'auto')
        
        if device_str == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                # RTX 5070 specific optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = self.config.get('hardware.cuda_benchmark', True)
                logger.info(f"Using CUDA with RTX 5070 optimizations for inference")
                logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                logger.info("CUDA not available, using CPU for inference")
        else:
            device = torch.device(device_str)
        
        return device
    
    def load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from checkpoint: {model_path}")
        else:
            self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded model weights from: {model_path}")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor.to(self.device)
    
    def postprocess_frame(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """Postprocess model output to frame."""
        # Remove batch dimension and move to CPU
        frame = frame_tensor.squeeze(0).cpu().numpy()
        
        # Denormalize from [0, 1] to [0, 255]
        frame = (frame * 255.0).astype(np.uint8)
        
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return frame
    
    def interpolate_frame_pair(self, frame1: np.ndarray, frame2: np.ndarray, 
                              num_frames: int = 1) -> List[np.ndarray]:
        """
        Interpolate frames between two input frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            num_frames: Number of frames to interpolate
            
        Returns:
            List of interpolated frames
        """
        # Preprocess frames
        frame1_tensor = self.preprocess_frame(frame1)
        frame2_tensor = self.preprocess_frame(frame2)
        
        interpolated_frames = []
        
        with torch.no_grad():
            if num_frames == 1:
                # Single frame interpolation
                predicted_tensor = self.model(frame1_tensor, frame2_tensor)
                predicted_frame = self.postprocess_frame(predicted_tensor)
                interpolated_frames.append(predicted_frame)
            else:
                # Multiple frame interpolation
                for i in range(1, num_frames + 1):
                    alpha = i / (num_frames + 1)
                    
                    # Create weighted combination of input frames
                    weighted_frame1 = frame1_tensor * (1 - alpha)
                    weighted_frame2 = frame2_tensor * alpha
                    combined_input = weighted_frame1 + weighted_frame2
                    
                    # Generate interpolated frame
                    predicted_tensor = self.model(frame1_tensor, frame2_tensor)
                    
                    # Apply temporal weighting
                    predicted_tensor = predicted_tensor * (1 - alpha) + predicted_tensor * alpha
                    
                    predicted_frame = self.postprocess_frame(predicted_tensor)
                    interpolated_frames.append(predicted_frame)
        
        return interpolated_frames
    
    def apply_temporal_smoothing(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply temporal smoothing to reduce flickering."""
        if not self.temporal_smoothing or len(frames) < 3:
            return frames
        
        smoothed_frames = []
        
        # First frame (no smoothing)
        smoothed_frames.append(frames[0])
        
        # Middle frames (apply smoothing)
        for i in range(1, len(frames) - 1):
            # Simple temporal smoothing using Gaussian blur
            smoothed = cv2.GaussianBlur(frames[i], (3, 3), 0)
            
            # Blend with original
            alpha = 0.7
            smoothed_frame = cv2.addWeighted(frames[i], alpha, smoothed, 1 - alpha, 0)
            smoothed_frames.append(smoothed_frame)
        
        # Last frame (no smoothing)
        smoothed_frames.append(frames[-1])
        
        return smoothed_frames
    
    def apply_post_processing(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Apply post-processing to improve frame quality."""
        if not self.post_processing:
            return frames
        
        processed_frames = []
        
        for frame in frames:
            # Apply sharpening filter
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(frame, -1, kernel)
            
            # Blend with original
            alpha = 0.3
            processed_frame = cv2.addWeighted(frame, 1 - alpha, sharpened, alpha, 0)
            
            # Apply slight denoising
            processed_frame = cv2.bilateralFilter(processed_frame, 9, 75, 75)
            
            processed_frames.append(processed_frame)
        
        return processed_frames
    
    def process_video(self, input_path: str, output_path: str):
        """
        Process entire video for frame interpolation.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
        """
        logger.info(f"Processing video: {input_path}")
        
        # Extract frames from input video
        frames = self.video_processor.extract_frames(input_path)
        
        if len(frames) < 2:
            raise ValueError("Video must contain at least 2 frames")
        
        logger.info(f"Extracted {len(frames)} frames from input video")
        
        # Generate interpolated frames
        interpolated_frames = []
        
        # Add first frame
        interpolated_frames.append(frames[0])
        
        # Process each consecutive frame pair
        for i in tqdm(range(len(frames) - 1), desc="Interpolating frames"):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Interpolate frames between the pair
            interpolated = self.interpolate_frame_pair(
                frame1, frame2, self.interpolation_factor
            )
            
            interpolated_frames.extend(interpolated)
            
            # Add the second frame
            interpolated_frames.append(frame2)
        
        logger.info(f"Generated {len(interpolated_frames)} total frames")
        
        # Apply post-processing
        if self.temporal_smoothing:
            interpolated_frames = self.apply_temporal_smoothing(interpolated_frames)
        
        if self.post_processing:
            interpolated_frames = self.apply_post_processing(interpolated_frames)
        
        # Calculate output FPS
        input_fps = self.config.data_config.get('input_fps', 15)
        target_fps = self.config.data_config.get('target_fps', 60)
        output_fps = input_fps * (self.interpolation_factor + 1)
        
        # Save output video
        self.video_processor.save_frames_as_video(
            interpolated_frames, output_path, output_fps
        )
        
        logger.info(f"Video processing completed. Output saved to: {output_path}")
        logger.info(f"Input FPS: {input_fps}, Output FPS: {output_fps}")
    
    def process_frame_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a sequence of frames for interpolation.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of interpolated frames
        """
        if len(frames) < 2:
            raise ValueError("At least 2 frames required for interpolation")
        
        interpolated_frames = []
        
        # Add first frame
        interpolated_frames.append(frames[0])
        
        # Process each consecutive frame pair
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Interpolate frames between the pair
            interpolated = self.interpolate_frame_pair(
                frame1, frame2, self.interpolation_factor
            )
            
            interpolated_frames.extend(interpolated)
            
            # Add the second frame
            interpolated_frames.append(frame2)
        
        # Apply post-processing
        if self.temporal_smoothing:
            interpolated_frames = self.apply_temporal_smoothing(interpolated_frames)
        
        if self.post_processing:
            interpolated_frames = self.apply_post_processing(interpolated_frames)
        
        return interpolated_frames
    
    def batch_process_videos(self, input_dir: str, output_dir: str):
        """
        Process multiple videos in batch.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory to save output videos
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(input_path.glob(f'**/*{ext}'))
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Process each video
        for video_file in video_files:
            try:
                output_file = output_path / f"interpolated_{video_file.name}"
                self.process_video(str(video_file), str(output_file))
            except Exception as e:
                logger.error(f"Error processing {video_file}: {e}")
                continue
        
        logger.info("Batch processing completed!")

class RealTimeFrameInterpolation:
    """Real-time frame interpolation for live video streams."""
    
    def __init__(self, config: Config, model_path: str):
        """
        Initialize real-time frame interpolation.
        
        Args:
            config: Configuration object
            model_path: Path to trained model
        """
        self.config = config
        self.device = self._setup_device()
        
        # Initialize model
        self.model = create_model(config.model_config)
        self.model.to(self.device)
        self.load_model(model_path)
        self.model.eval()
        
        # Frame buffer for temporal consistency
        self.frame_buffer = []
        self.buffer_size = 3
        
        logger.info("Real-time frame interpolation initialized")
    
    def _setup_device(self) -> torch.device:
        """Setup device for real-time processing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self, model_path: str):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for real-time interpolation.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        # Add frame to buffer
        self.frame_buffer.append(frame.copy())
        
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # If we have enough frames, interpolate
        if len(self.frame_buffer) >= 2:
            frame1 = self.frame_buffer[-2]
            frame2 = self.frame_buffer[-1]
            
            # Interpolate between frames
            interpolated = self._interpolate_frames(frame1, frame2)
            return interpolated
        
        return frame
    
    def _interpolate_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Interpolate between two frames."""
        # Preprocess frames
        frame1_tensor = self._preprocess_frame(frame1)
        frame2_tensor = self._preprocess_frame(frame2)
        
        with torch.no_grad():
            predicted_tensor = self.model(frame1_tensor, frame2_tensor)
            predicted_frame = self._postprocess_frame(predicted_tensor)
        
        return predicted_frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return frame_tensor.to(self.device)
    
    def _postprocess_frame(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """Postprocess model output."""
        frame = frame_tensor.squeeze(0).cpu().numpy()
        frame = (frame * 255.0).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
