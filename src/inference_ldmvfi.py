"""
LDMVFI-based inference for frame interpolation.
Uses diffusion sampling (DDPM/DDIM) for frame interpolation.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from typing import Optional, Union

from src.utils.ldm_util import instantiate_from_config
from src.models.diffusion.ddim import DDIMSampler


class LDMVFIInference:
    """
    Inference class for LDMVFI frame interpolation.
    Uses latent diffusion model with DDPM/DDIM sampling.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str, 
                 use_ddim: bool = True, ddim_steps: int = 200, ddim_eta: float = 1.0,
                 device: Optional[str] = None):
        """
        Initialize LDMVFI inference.
        
        Args:
            config_path: Path to model config YAML file
            checkpoint_path: Path to model checkpoint
            use_ddim: Use DDIM sampling (faster) instead of DDPM
            ddim_steps: Number of DDIM sampling steps
            ddim_eta: DDIM eta parameter (0.0 = DDPM, 1.0 = DDIM)
            device: Device to use ('cuda' or 'cpu')
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load config
        self.config = OmegaConf.load(config_path)
        
        # Initialize model
        self.model = instantiate_from_config(self.config.model)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup sampler
        if self.use_ddim:
            self.sampler = DDIMSampler(self.model)
            self.sample_func = lambda c, shape, **kwargs: self.sampler.sample(
                S=self.ddim_steps, 
                conditioning=c, 
                batch_size=1,
                shape=shape,
                eta=self.ddim_eta,
                verbose=False,
                **kwargs
            )
        else:
            # Use DDPM sampling
            self.sample_func = lambda c, shape, **kwargs: self.model.sample_ddpm(
                conditioning=c,
                batch_size=1,
                shape=shape,
                return_intermediates=False,
                verbose=False,
                **kwargs
            )
        
        # Transform for input images: PIL Image -> Tensor (normalized to [-1, 1])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [0,1] -> [-1,1]
        ])
        
        print(f"LDMVFI model loaded on {self.device}")
        print(f"Using {'DDIM' if self.use_ddim else 'DDPM'} sampling")
    
    def interpolate_frame(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Interpolate a frame between frame1 and frame2.
        
        Args:
            frame1: Previous frame (H, W, 3) in RGB, range [0, 255]
            frame2: Next frame (H, W, 3) in RGB, range [0, 255]
        
        Returns:
            Interpolated frame (H, W, 3) in RGB, range [0, 255]
        """
        with torch.no_grad():
            with self.model.ema_scope():
                # Convert to PIL Images
                frame1_pil = Image.fromarray(frame1.astype(np.uint8))
                frame2_pil = Image.fromarray(frame2.astype(np.uint8))
                
                # Transform to tensors (normalized to [-1, 1])
                frame1_tensor = self.transform(frame1_pil).unsqueeze(0).to(self.device)
                frame2_tensor = self.transform(frame2_pil).unsqueeze(0).to(self.device)
                
                # Form condition tensor
                xc = {
                    'prev_frame': frame1_tensor,
                    'next_frame': frame2_tensor
                }
                
                # Get learned conditioning
                c, phi_prev_list, phi_next_list = self.model.get_learned_conditioning(xc)
                
                # Define shape of latent representation
                shape = (self.model.channels, c.shape[2], c.shape[3])
                
                # Run sampling
                out = self.sample_func(
                    c,
                    shape,
                    x_T=None
                )
                
                # Handle tuple output (DDIM returns (samples, intermediates))
                if isinstance(out, tuple):
                    out = out[0]
                
                # Decode from latent space to pixel space
                out = self.model.decode_first_stage(out, xc, phi_prev_list, phi_next_list)
                
                # Clamp to [-1, 1] range
                out = torch.clamp(out, min=-1., max=1.)
                
                # Convert back to numpy: [-1, 1] -> [0, 255]
                out = ((out + 1.0) / 2.0)  # [-1, 1] -> [0, 1]
                out = out.squeeze(0).cpu().numpy()  # Remove batch dimension
                out = out.transpose(1, 2, 0)  # CHW -> HWC
                out = (out * 255).clip(0, 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV compatibility (if needed)
                # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                
                return out
    
    def interpolate_video(self, input_path: str, output_path: str, 
                         fps_multiplier: int = 2):
        """
        Interpolate frames in a video.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            fps_multiplier: FPS multiplier (e.g., 2 = 2x FPS)
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps * fps_multiplier
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        # Convert BGR to RGB
        prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        
        # Write first frame
        out.write(prev_frame)
        
        frame_count = 1
        
        # Process video
        while True:
            ret, next_frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            next_frame_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
            
            # Interpolate frames
            num_interpolated = fps_multiplier - 1
            for i in range(num_interpolated):
                # Interpolate frame
                interpolated_rgb = self.interpolate_frame(prev_frame_rgb, next_frame_rgb)
                
                # Convert RGB to BGR for writing
                interpolated_bgr = cv2.cvtColor(interpolated_rgb, cv2.COLOR_RGB2BGR)
                out.write(interpolated_bgr)
            
            # Write original next frame
            out.write(next_frame)
            
            prev_frame_rgb = next_frame_rgb
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Video interpolation complete: {output_path}")


def load_model_for_inference(config_path: str, checkpoint_path: str, 
                            use_ddim: bool = True, device: Optional[str] = None):
    """
    Convenience function to load LDMVFI model for inference.
    
    Args:
        config_path: Path to model config YAML
        checkpoint_path: Path to model checkpoint
        use_ddim: Use DDIM sampling
        device: Device to use
    
    Returns:
        LDMVFIInference instance
    """
    return LDMVFIInference(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        use_ddim=use_ddim,
        device=device
    )
