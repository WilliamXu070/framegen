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
from src.data.ucf101_processor import UCF101Processor
from src.utils.logger import setup_logger
from src.utils.video_utils import VideoProcessor

def load_ucf101_demo_frames(config, num_videos=3, frames_per_video=5):
    """Load demo frames from UCF101 dataset."""
    logger = setup_logger()
    logger.info("Loading UCF101 dataset for demo...")
    
    # Initialize UCF101 processor
    ucf101_processor = UCF101Processor(config.config)
    
    # Check if dataset is processed
    if not ucf101_processor.is_processed():
        logger.info("UCF101 dataset not processed. Processing dataset...")
        if not ucf101_processor.process_dataset():
            logger.error("Failed to process UCF101 dataset. Falling back to synthetic frames.")
            return create_synthetic_fallback_frames(num_videos * frames_per_video)
    
    # Get dataset info
    dataset_info = ucf101_processor.get_processed_info()
    logger.info(f"UCF101 dataset info: {dataset_info}")
    
    # Load frames from processed dataset
    demo_frames = []
    train_dir = ucf101_processor.train_dir
    video_dirs = list(train_dir.glob('video_*'))
    
    if not video_dirs:
        logger.error("No processed videos found in UCF101 dataset. Falling back to synthetic frames.")
        return create_synthetic_fallback_frames(num_videos * frames_per_video)
    
    # Select random videos for demo
    selected_videos = random.sample(video_dirs, min(num_videos, len(video_dirs)))
    
    for video_dir in selected_videos:
        logger.info(f"Processing video: {video_dir.name}")
        
        # Get frame files
        frame_files = sorted(list(video_dir.glob('frame_*.jpg')))
        
        if len(frame_files) < frames_per_video:
            logger.warning(f"Video {video_dir.name} has only {len(frame_files)} frames, skipping")
            continue
        
        # Sample frames evenly
        frame_indices = np.linspace(0, len(frame_files) - 1, frames_per_video, dtype=int)
        
        for frame_idx in frame_indices:
            frame_path = frame_files[frame_idx]
            frame = cv2.imread(str(frame_path))
            
            if frame is not None:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                demo_frames.append(frame)
            else:
                logger.warning(f"Failed to read frame {frame_path}")
    
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

def run_test_mode(config, args):
    """Run testing mode for video generation with frame interpolation."""
    logger = setup_logger()
    logger.info("Starting testing mode for video generation")
    
    # Check if model exists
    model_path = args.model or 'models/best_model.pth'
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train the model first or specify --model path")
        return
    
    # Check if test video is provided
    if not args.test_video:
        logger.error("Test video path is required for testing mode. Use --test-video path/to/video.mp4")
        return
    
    if not os.path.exists(args.test_video):
        logger.error(f"Test video not found at {args.test_video}")
        return
    
    # Create output directory
    output_dir = Path(args.test_output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize inference
    logger.info("Loading trained model...")
    inference = FrameInterpolationInference(config, model_path)
    
    # Extract frames from test video
    logger.info(f"Extracting frames from test video: {args.test_video}")
    video_processor = VideoProcessor()
    original_frames = video_processor.extract_frames(args.test_video)
    
    if len(original_frames) < 2:
        logger.error("Test video must contain at least 2 frames")
        return
    
    logger.info(f"Extracted {len(original_frames)} frames from test video")
    
    # Generate interpolated frames
    logger.info(f"Generating interpolated frames with {args.test_fps_multiplier}x FPS enhancement...")
    interpolated_frames = []
    
    # Add first frame
    interpolated_frames.append(original_frames[0])
    
    # Process each consecutive frame pair
    for i in range(len(original_frames) - 1):
        frame1 = original_frames[i]
        frame2 = original_frames[i + 1]
        
        # Interpolate frames between the pair (1 frame for 2x FPS)
        interpolated = inference.interpolate_frame_pair(frame1, frame2, 1)
        interpolated_frames.extend(interpolated)
        
        # Add the second frame
        interpolated_frames.append(frame2)
    
    logger.info(f"Generated {len(interpolated_frames)} total frames")
    
    # Get original video properties
    cap = cv2.VideoCapture(args.test_video)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Calculate new FPS
    new_fps = original_fps * args.test_fps_multiplier
    
    # Save original video (resampled to show comparison)
    original_video_path = output_dir / "original_video.mp4"
    video_processor.save_frames_as_video(original_frames, str(original_video_path), original_fps)
    logger.info(f"Original video saved to: {original_video_path}")
    
    # Save enhanced video with interpolated frames
    enhanced_video_path = output_dir / "enhanced_video.mp4"
    video_processor.save_frames_as_video(interpolated_frames, str(enhanced_video_path), new_fps)
    logger.info(f"Enhanced video saved to: {enhanced_video_path}")
    
    # Save frame-by-frame comparison
    save_frame_comparison(original_frames, interpolated_frames, output_dir)
    
    # Print results
    print(f"\n{'='*60}")
    print("TEST MODE RESULTS")
    print(f"{'='*60}")
    print(f"Original video: {args.test_video}")
    print(f"Original frames: {len(original_frames)}")
    print(f"Enhanced frames: {len(interpolated_frames)}")
    print(f"Original FPS: {original_fps:.2f}")
    print(f"Enhanced FPS: {new_fps:.2f}")
    print(f"FPS multiplier: {args.test_fps_multiplier}x")
    print(f"Output directory: {output_dir}")
    print(f"Original video saved: {original_video_path}")
    print(f"Enhanced video saved: {enhanced_video_path}")
    print(f"{'='*60}")
    
    logger.info("Testing mode completed successfully!")

def save_frame_comparison(original_frames, interpolated_frames, output_dir):
    """Save frame-by-frame comparison images."""
    logger = setup_logger()
    logger.info("Saving frame-by-frame comparison...")
    
    comparison_dir = output_dir / "frame_comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    # Save original frames
    original_dir = comparison_dir / "original"
    original_dir.mkdir(exist_ok=True)
    
    for i, frame in enumerate(original_frames):
        frame_path = original_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
    
    # Save interpolated frames
    interpolated_dir = comparison_dir / "interpolated"
    interpolated_dir.mkdir(exist_ok=True)
    
    for i, frame in enumerate(interpolated_frames):
        frame_path = interpolated_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
    
    logger.info(f"Frame comparison saved to: {comparison_dir}")

def main():
    """Main entry point for the frame generation application."""
    parser = argparse.ArgumentParser(description='Frame Generation Application')
    parser.add_argument('--mode', choices=['train', 'inference', 'demo', 'test'], required=True,
                       help='Mode: train the model, run inference, run visual demonstration, or test video generation')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, help='Input video path for inference')
    parser.add_argument('--output', type=str, help='Output video path for inference')
    parser.add_argument('--model', type=str, help='Path to trained model for inference')
    parser.add_argument('--demo-frames', type=int, default=5, help='Number of demo frames to extract from UCF101')
    parser.add_argument('--interpolation-factor', type=int, default=3, help='Number of frames to interpolate between each pair')
    parser.add_argument('--no-display', action='store_true', help='Skip matplotlib display and only show console output')
    parser.add_argument('--num-videos', type=int, default=2, help='Number of UCF101 videos to sample from')
    parser.add_argument('--light-loading', action='store_true', help='Use light loading mode with only 100 training videos for faster testing')
    parser.add_argument('--test-video', type=str, help='Path to test video for testing mode')
    parser.add_argument('--test-output-dir', type=str, default='demo', help='Output directory for test videos')
    parser.add_argument('--test-fps-multiplier', type=int, default=2, help='FPS multiplier for test videos (e.g., 2 for 2x FPS)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger()
    logger.info(f"Starting Frame Generation Application in {args.mode} mode")
    
    # Load configuration
    config = Config(args.config)
    
    if args.mode == 'train':
        trainer = FrameInterpolationTrainer(config, light_loading=args.light_loading)
        trainer.train()
    elif args.mode == 'inference':
        if not args.input or not args.output:
            logger.error("Input and output paths are required for inference mode")
            sys.exit(1)
        
        inference = FrameInterpolationInference(config, args.model)
        inference.process_video(args.input, args.output)
    elif args.mode == 'demo':
        run_visual_demo(config, args)
    elif args.mode == 'test':
        run_test_mode(config, args)

if __name__ == "__main__":
    main()
