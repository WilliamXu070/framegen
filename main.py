#!/usr/bin/env python3
"""
Frame Generation Application - Main Entry Point
A Python application for generating intermediate frames in low frame rate videos
using advanced computer vision and machine learning techniques.
"""

import argparse
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import torch
import random

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import Config
from src.trainer import FrameInterpolationTrainer
from src.inference import FrameInterpolationInference
from src.data.ucf101_dataset import UCF101DatasetManager
from src.data.ucf101_dataset_loader import create_ucf101_data_loaders
from src.utils.logger import setup_logger

def load_ucf101_demo_frames(config, num_videos=3, frames_per_video=5):
    """Load demo frames from UCF101 dataset."""
    logger = setup_logger()
    logger.info("Loading UCF101 dataset for demo...")
    
    # Initialize UCF101 dataset manager
    ucf101_manager = UCF101DatasetManager(config.config)
    
    # Check if dataset is downloaded
    if not ucf101_manager.is_dataset_downloaded():
        logger.info("UCF101 dataset not found. Downloading from Hugging Face...")
        if not ucf101_manager.prepare_dataset():
            logger.error("Failed to download UCF101 dataset. Falling back to synthetic frames.")
            return create_synthetic_fallback_frames(num_videos * frames_per_video)
    
    # Validate dataset
    if not ucf101_manager.validate_dataset():
        logger.error("UCF101 dataset validation failed. Falling back to synthetic frames.")
        return create_synthetic_fallback_frames(num_videos * frames_per_video)
    
    # Get dataset info
    dataset_info = ucf101_manager.get_dataset_info()
    logger.info(f"UCF101 dataset info: {dataset_info}")
    
    # Load a few sample videos
    demo_frames = []
    video_files = list(ucf101_manager.raw_dir.glob('*.avi'))
    
    if not video_files:
        logger.error("No video files found in UCF101 dataset. Falling back to synthetic frames.")
        return create_synthetic_fallback_frames(num_videos * frames_per_video)
    
    # Select random videos for demo
    selected_videos = random.sample(video_files, min(num_videos, len(video_files)))
    
    for video_path in selected_videos:
        logger.info(f"Processing video: {video_path.name}")
        
        # Extract frames from video
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count < frames_per_video:
            logger.warning(f"Video {video_path.name} has only {frame_count} frames, skipping")
            cap.release()
            continue
        
        # Sample frames evenly
        frame_indices = np.linspace(0, frame_count - 1, frames_per_video, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame to match config
                frame_size = tuple(config.data_config.get('frame_size', [256, 256]))
                frame = cv2.resize(frame, frame_size)
                demo_frames.append(frame)
            else:
                logger.warning(f"Failed to read frame {frame_idx} from {video_path.name}")
        
        cap.release()
    
    if not demo_frames:
        logger.error("No frames extracted from UCF101 dataset. Falling back to synthetic frames.")
        return create_synthetic_fallback_frames(num_videos * frames_per_video)
    
    logger.info(f"Successfully loaded {len(demo_frames)} frames from UCF101 dataset")
    return demo_frames

def create_synthetic_fallback_frames(num_frames=5, frame_size=(256, 256)):
    """Create synthetic fallback frames if UCF101 fails."""
    logger = setup_logger()
    logger.warning("Creating synthetic fallback frames...")
    
    frames = []
    for i in range(num_frames):
        # Create a black frame
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        # Create a moving circle
        center_x = int(frame_size[0] * 0.2 + (frame_size[0] * 0.6) * i / (num_frames - 1))
        center_y = int(frame_size[1] * 0.5 + 50 * np.sin(2 * np.pi * i / num_frames))
        radius = 30
        
        # Color changes over time
        color = (
            int(255 * i / num_frames),
            int(255 * (1 - i / num_frames)),
            int(128 + 127 * np.sin(2 * np.pi * i / num_frames))
        )
        
        # Draw circle
        cv2.circle(frame, (center_x, center_y), radius, color, -1)
        
        # Add some text
        text = f"Frame {i+1}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add some noise for realism
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        frames.append(frame)
    
    return frames

def run_visual_demo(config, args):
    """Run visual demonstration of the frame interpolation model."""
    logger = setup_logger()
    logger.info("Starting visual demonstration of frame interpolation model")
    
    # Check if model exists
    model_path = args.model or 'models/best_model.pth'
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train the model first or specify --model path")
        return
    
    # Load demo frames from UCF101 dataset
    logger.info(f"Loading {args.demo_frames} demo frames from UCF101 dataset...")
    demo_frames = load_ucf101_demo_frames(config, num_videos=args.num_videos, frames_per_video=args.demo_frames)
    
    # Initialize inference
    logger.info("Loading trained model...")
    inference = FrameInterpolationInference(config, model_path)
    
    # Generate interpolated frames
    logger.info(f"Generating interpolated frames with factor {args.interpolation_factor}...")
    interpolated_frames = []
    
    # Add first frame
    interpolated_frames.append(demo_frames[0])
    
    # Process each consecutive frame pair
    for i in range(len(demo_frames) - 1):
        frame1 = demo_frames[i]
        frame2 = demo_frames[i + 1]
        
        # Interpolate frames between the pair
        interpolated = inference.interpolate_frame_pair(frame1, frame2, args.interpolation_factor)
        interpolated_frames.extend(interpolated)
        
        # Add the second frame
        interpolated_frames.append(frame2)
    
    logger.info(f"Generated {len(interpolated_frames)} total frames")
    
    # Show frame statistics
    print(f"\n{'='*60}")
    print("FRAME INTERPOLATION DEMO RESULTS")
    print(f"{'='*60}")
    print(f"Original frames: {len(demo_frames)}")
    print(f"Interpolated frames: {len(interpolated_frames)}")
    print(f"Interpolation factor: {args.interpolation_factor}")
    print(f"Total frames generated: {len(interpolated_frames)}")
    print(f"Frame size: {demo_frames[0].shape}")
    print(f"Model device: {inference.device}")
    print(f"{'='*60}")
    
    # Create visual comparison (if not disabled)
    if not args.no_display:
        try:
            create_comparison_visualization(demo_frames, interpolated_frames, args.interpolation_factor)
            create_animated_comparison(demo_frames, interpolated_frames)
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visual display.")
        except Exception as e:
            logger.warning(f"Visual display failed: {e}")
    else:
        logger.info("Visual display skipped (--no-display flag)")
    
    # Save sample frames as numpy arrays for inspection
    save_demo_results(demo_frames, interpolated_frames, args.interpolation_factor)
    
    logger.info("Visual demonstration completed!")

def create_comparison_visualization(original_frames, interpolated_frames, interpolation_factor):
    """Create a side-by-side comparison visualization."""
    logger = setup_logger()
    logger.info("Creating comparison visualization...")
    
    # Calculate grid size
    num_original = len(original_frames)
    num_interpolated = len(interpolated_frames)
    
    # Create figure
    fig, axes = plt.subplots(2, max(num_original, num_interpolated), figsize=(20, 8))
    if num_original == 1:
        axes = axes.reshape(2, 1)
    
    # Plot original frames
    for i, frame in enumerate(original_frames):
        if i < axes.shape[1]:
            # Convert BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(frame_rgb)
            axes[0, i].set_title(f'Original Frame {i+1}')
            axes[0, i].axis('off')
    
    # Hide unused subplots for original frames
    for i in range(num_original, axes.shape[1]):
        axes[0, i].axis('off')
    
    # Plot interpolated frames
    for i, frame in enumerate(interpolated_frames):
        if i < axes.shape[1]:
            # Convert BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[1, i].imshow(frame_rgb)
            axes[1, i].set_title(f'Interpolated Frame {i+1}')
            axes[1, i].axis('off')
    
    # Hide unused subplots for interpolated frames
    for i in range(num_interpolated, axes.shape[1]):
        axes[1, i].axis('off')
    
    plt.suptitle(f'Frame Interpolation Demo (Factor: {interpolation_factor})', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    logger.info("Comparison visualization displayed")

def create_animated_comparison(original_frames, interpolated_frames):
    """Create an animated comparison showing the interpolation effect."""
    logger = setup_logger()
    logger.info("Creating animated comparison...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert frames to RGB
    original_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in original_frames]
    interpolated_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in interpolated_frames]
    
    # Initialize plots
    im1 = ax1.imshow(original_rgb[0])
    im2 = ax2.imshow(interpolated_rgb[0])
    
    ax1.set_title('Original Frames (Low FPS)')
    ax2.set_title('Interpolated Frames (High FPS)')
    ax1.axis('off')
    ax2.axis('off')
    
    def animate(frame_idx):
        # Cycle through original frames (slower)
        orig_idx = frame_idx % len(original_rgb)
        im1.set_array(original_rgb[orig_idx])
        
        # Cycle through interpolated frames (faster)
        interp_idx = frame_idx % len(interpolated_rgb)
        im2.set_array(interpolated_rgb[interp_idx])
        
        return im1, im2
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(interpolated_frames) * 2, 
                                 interval=200, blit=True, repeat=True)
    
    plt.suptitle('Frame Interpolation Animation Demo', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    logger.info("Animated comparison displayed")

def save_demo_results(original_frames, interpolated_frames, interpolation_factor):
    """Save demo results as numpy arrays for inspection."""
    logger = setup_logger()
    logger.info("Saving demo results...")
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save original frames
    original_array = np.array(original_frames)
    np.save(output_dir / "original_frames.npy", original_array)
    
    # Save interpolated frames
    interpolated_array = np.array(interpolated_frames)
    np.save(output_dir / "interpolated_frames.npy", interpolated_array)
    
    # Save metadata
    metadata = {
        'interpolation_factor': interpolation_factor,
        'original_frames_count': len(original_frames),
        'interpolated_frames_count': len(interpolated_frames),
        'frame_shape': original_frames[0].shape,
        'total_frames': len(interpolated_frames)
    }
    
    import json
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Demo results saved to {output_dir}")
    logger.info(f"Original frames shape: {original_array.shape}")
    logger.info(f"Interpolated frames shape: {interpolated_array.shape}")

def main():
    """Main entry point for the frame generation application."""
    parser = argparse.ArgumentParser(description='Frame Generation Application')
    parser.add_argument('--mode', choices=['train', 'inference', 'demo'], required=True,
                       help='Mode: train the model, run inference, or run visual demonstration')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Input video path for inference')
    parser.add_argument('--output', type=str, help='Output video path for inference')
    parser.add_argument('--model', type=str, help='Path to trained model for inference')
    parser.add_argument('--demo-frames', type=int, default=5, help='Number of demo frames to extract from UCF101')
    parser.add_argument('--interpolation-factor', type=int, default=3, help='Number of frames to interpolate between each pair')
    parser.add_argument('--no-display', action='store_true', help='Skip matplotlib display and only show console output')
    parser.add_argument('--num-videos', type=int, default=2, help='Number of UCF101 videos to sample from')
    
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
    elif args.mode == 'demo':
        run_visual_demo(config, args)

if __name__ == "__main__":
    main()
