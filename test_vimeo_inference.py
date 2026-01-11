#!/usr/bin/env python3
"""
Test inference on Vimeo triplet dataset and create interpolated video.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import Config
from src.models.frame_interpolation import create_model
from src.utils.logger import setup_logger

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images (simplified)."""
    from skimage.metrics import structural_similarity as ssim
    return ssim(img1, img2, multichannel=True, channel_axis=2, data_range=255)

def load_vimeo_triplet(triplet_path):
    """Load a Vimeo triplet (3 frames)."""
    im1 = cv2.imread(str(triplet_path / "im1.png"))
    im2 = cv2.imread(str(triplet_path / "im2.png"))  # Ground truth middle frame
    im3 = cv2.imread(str(triplet_path / "im3.png"))

    # Convert BGR to RGB
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)

    return im1, im2, im3

def preprocess_frame(frame):
    """Preprocess frame for model input."""
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    # Convert to tensor (C, H, W)
    frame = torch.from_numpy(frame).permute(2, 0, 1)
    return frame

def postprocess_frame(tensor):
    """Convert model output back to image."""
    # Convert to numpy (H, W, C)
    frame = tensor.permute(1, 2, 0).cpu().numpy()
    # Clip and convert to uint8
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame

def create_comparison_video(im1, im2_gt, im2_pred, im3, output_path, fps=5):
    """Create a comparison video showing: original frames, interpolated, ground truth."""
    height, width = im1.shape[:2]

    # Create side-by-side comparison
    # Top row: im1 | interpolated | im3
    # Bottom row: ground truth im2 | difference map

    comparison_height = height * 2
    comparison_width = width * 3

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (comparison_width, comparison_height))

    # Create frames for video
    frames_to_show = [
        ("Original Sequence", [im1, im2_pred, im3]),
        ("With Ground Truth", [im1, im2_gt, im3]),
    ]

    for title, frames in frames_to_show:
        # Create comparison frame
        comparison = np.zeros((comparison_height, comparison_width, 3), dtype=np.uint8)

        # Top row: im1, interpolated, im3
        comparison[0:height, 0:width] = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
        comparison[0:height, width:width*2] = cv2.cvtColor(im2_pred, cv2.COLOR_RGB2BGR)
        comparison[0:height, width*2:width*3] = cv2.cvtColor(im3, cv2.COLOR_RGB2BGR)

        # Bottom row: ground truth and difference
        comparison[height:height*2, 0:width] = cv2.cvtColor(im2_gt, cv2.COLOR_RGB2BGR)

        # Difference map
        diff = np.abs(im2_gt.astype(float) - im2_pred.astype(float))
        diff = (diff * 3).clip(0, 255).astype(np.uint8)  # Amplify differences
        comparison[height:height*2, width:width*2] = cv2.cvtColor(diff, cv2.COLOR_RGB2BGR)

        # Error heatmap
        error = np.mean(diff, axis=2)
        heatmap = cv2.applyColorMap(error.astype(np.uint8), cv2.COLORMAP_JET)
        comparison[height:height*2, width*2:width*3] = heatmap

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Frame 1", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Interpolated", (width + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Frame 3", (width*2 + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Ground Truth", (10, height + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Difference", (width + 10, height + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Error Heatmap", (width*2 + 10, height + 30), font, 1, (255, 255, 255), 2)

        # Write frame multiple times to make it visible
        for _ in range(fps * 2):  # Show each for 2 seconds
            out.write(comparison)

    out.release()
    print(f"Comparison video saved to: {output_path}")

def test_vimeo_triplet():
    """Test model on Vimeo triplet dataset."""
    logger = setup_logger()
    logger.info("Testing on Vimeo Triplet Dataset")

    # Setup
    config = Config('configs/default.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_path = 'models/best_model.pth'
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = create_model(config.model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully on {device}")

    # Get test triplets
    vimeo_dir = Path('data/vimeo_triplet')
    test_list_path = vimeo_dir / 'tri_testlist.txt'

    with open(test_list_path, 'r') as f:
        test_triplets = [line.strip() for line in f.readlines()]

    # Select first triplet for testing
    triplet_name = test_triplets[0]
    triplet_path = vimeo_dir / 'sequences' / triplet_name

    logger.info(f"Testing on triplet: {triplet_name}")

    # Load frames
    im1, im2_gt, im3 = load_vimeo_triplet(triplet_path)
    logger.info(f"Loaded frames: {im1.shape}")

    # Preprocess
    frame1_tensor = preprocess_frame(im1).unsqueeze(0).to(device)
    frame3_tensor = preprocess_frame(im3).unsqueeze(0).to(device)

    # Run inference
    logger.info("Running inference...")
    with torch.no_grad():
        im2_pred_tensor = model(frame1_tensor, frame3_tensor, alpha=0.5)

    # Postprocess
    im2_pred = postprocess_frame(im2_pred_tensor.squeeze(0))

    logger.info(f"Interpolated frame shape: {im2_pred.shape}")

    # Calculate metrics
    psnr = calculate_psnr(im2_gt, im2_pred)
    ssim = calculate_ssim(im2_gt, im2_pred)

    logger.info("="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Triplet: {triplet_name}")
    logger.info(f"PSNR: {psnr:.2f} dB")
    logger.info(f"SSIM: {ssim:.4f}")
    logger.info("="*60)

    # Save outputs
    output_dir = Path('output/vimeo_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual frames
    Image.fromarray(im1).save(output_dir / 'frame1_input.png')
    Image.fromarray(im2_gt).save(output_dir / 'frame2_ground_truth.png')
    Image.fromarray(im2_pred).save(output_dir / 'frame2_interpolated.png')
    Image.fromarray(im3).save(output_dir / 'frame3_input.png')

    # Save difference map
    diff = np.abs(im2_gt.astype(float) - im2_pred.astype(float))
    diff = (diff * 3).clip(0, 255).astype(np.uint8)
    Image.fromarray(diff).save(output_dir / 'difference_map.png')

    logger.info(f"Saved output frames to: {output_dir}")

    # Create comparison video
    video_path = output_dir / 'comparison_video.mp4'
    create_comparison_video(im1, im2_gt, im2_pred, im3, video_path)

    logger.info("✅ Test completed successfully!")

    return {
        'psnr': psnr,
        'ssim': ssim,
        'triplet': triplet_name,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    results = test_vimeo_triplet()
