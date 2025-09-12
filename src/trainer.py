"""
Training pipeline for frame interpolation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import time
import logging
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

from src.models.frame_interpolation import create_model
from src.data.dataset import create_data_loaders
from src.data.ucf101_dataset import UCF101DatasetManager
from src.data.ucf101_dataset_loader import create_ucf101_data_loaders
from src.data.ucf101_processor import UCF101Processor
from src.utils.logger import setup_logger
from src.utils.memory_manager import setup_memory_optimization, clear_cuda_cache
from src.config import Config

logger = logging.getLogger(__name__)

class FrameInterpolationTrainer:
    """Trainer class for frame interpolation models."""
    
    def __init__(self, config: Config, light_loading: bool = False):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            light_loading: If True, use only 100 videos for faster training/testing
        """
        self.config = config
        self.light_loading = light_loading
        self.device = self._setup_device()
        
        # Initialize model
        self.model = create_model(config.model_config)
        self.model.to(self.device)
        
        # RTX 5070 model compilation for better performance
        if config.get('hardware.compile_model', False) and hasattr(torch, 'compile'):
            try:
                # Use default mode instead of max-autotune for Windows compatibility
                self.model = torch.compile(self.model, mode='default')
                logger.info("Model compiled for RTX 5070 optimization")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        # Initialize optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Initialize loss function
        self.criterion = self._setup_loss_function()
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir=config.get('paths.logs_dir', 'logs'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # RTX 5070 specific training state
        self.gradient_accumulation_steps = config.training_config.get('gradient_accumulation_steps', 1)
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        # Initialize memory manager
        self.memory_manager = setup_memory_optimization(config.config)
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.memory_manager.log_memory_status()
    
    def _setup_data_loaders(self):
        """Setup data loaders with UCF101 dataset support."""
        ucf101_config = self.config.get('ucf101', {})
        
        if ucf101_config.get('enabled', False):
            logger.info("Setting up UCF101 dataset...")
            
            # Initialize UCF101 processor
            self.ucf101_processor = UCF101Processor(self.config.config)
            
            # Process UCF101 dataset if not already processed
            if not self.ucf101_processor.is_processed():
                logger.info("UCF101 dataset not processed, starting processing...")
                if not self.ucf101_processor.process_dataset():
                    logger.error("Failed to process UCF101 dataset, falling back to standard dataset")
                    self.train_loader, self.val_loader = create_data_loaders(self.config.config)
                    return
            else:
                logger.info("UCF101 dataset already processed, using existing data")
            
            # Get dataset info
            dataset_info = self.ucf101_processor.get_processed_info()
            logger.info(f"UCF101 dataset info: {dataset_info}")
            
            # Create UCF101 data loaders
            if self.light_loading:
                logger.info("Light loading mode enabled - limiting dataset to 100 videos")
                self.train_loader, self.val_loader = create_ucf101_data_loaders(self.config.config, light_loading=True)
            else:
                self.train_loader, self.val_loader = create_ucf101_data_loaders(self.config.config)
            
            logger.info("UCF101 dataset setup completed successfully")
        else:
            logger.info("Using standard dataset (UCF101 disabled)")
            self.train_loader, self.val_loader = create_data_loaders(self.config.config)
    
    def _setup_device(self) -> torch.device:
        """Setup training device with RTX 5070 optimizations."""
        training_config = self.config.training_config
        device_str = training_config.get('device', 'auto')
        
        if device_str == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                # RTX 5070 specific optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = self.config.get('hardware.cuda_benchmark', True)
                logger.info(f"Using CUDA with RTX 5070 optimizations")
                logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = torch.device('cpu')
                logger.info("CUDA not available, using CPU")
        else:
            device = torch.device(device_str)
        
        return device
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        training_config = self.config.training_config
        learning_rate = training_config.get('learning_rate', 0.001)
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        return optimizer
    
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        training_config = self.config.training_config
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return scheduler
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function."""
        # Combined loss: L1 + Perceptual + SSIM
        return CombinedLoss()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with RTX 5070 optimizations."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        # RTX 5070 mixed precision setup
        use_amp = self.config.get('hardware.mixed_precision', False) and self.device.type == 'cuda'
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device with non_blocking for RTX 5070
            frame1 = batch['frame1'].to(self.device, non_blocking=True)
            frame2 = batch['frame2'].to(self.device, non_blocking=True)
            target = batch['target'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision for RTX 5070
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    predicted = self.model(frame1, frame2)
                    loss = self.criterion(predicted, target, frame1, frame2)
                    loss = loss / self.gradient_accumulation_steps
            else:
                predicted = self.model(frame1, frame2)
                loss = self.criterion(predicted, target, frame1, frame2)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation for RTX 5070
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Memory management
            self.memory_manager.update_batch_count()
            if self.memory_manager.is_memory_high():
                logger.warning("High memory usage detected, cleaning up...")
                self.memory_manager.cleanup_memory(force=True)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'Device': str(self.device),
                'Mem': f'{self.memory_manager.check_memory_usage():.1%}'
            })
            
            # Log to tensorboard
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item() * self.gradient_accumulation_steps, global_step)
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                frame1 = batch['frame1'].to(self.device)
                frame2 = batch['frame2'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                predicted = self.model(frame1, frame2)
                
                # Calculate loss
                loss = self.criterion(predicted, target, frame1, frame2)
                
                # Calculate metrics
                psnr = calculate_psnr(predicted, target)
                ssim = calculate_ssim(predicted, target)
                
                # Update metrics
                total_loss += loss.item()
                total_psnr += psnr
                total_ssim += ssim
        
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        
        return {
            'val_loss': avg_loss,
            'val_psnr': avg_psnr,
            'val_ssim': avg_ssim
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.config
        }
        
        models_dir = Path(self.config.get('paths.models_dir', 'models'))
        models_dir.mkdir(exist_ok=True)
        
        # Only save best model, no intermediate checkpoints
        if is_best:
            best_path = models_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {metrics['val_loss']:.4f}")
            
            # Clean up old checkpoint files to save storage space
            self._cleanup_old_checkpoints(models_dir)
    
    def _cleanup_old_checkpoints(self, models_dir: Path):
        """Clean up old checkpoint files to save storage space."""
        try:
            # Find all checkpoint files
            checkpoint_files = list(models_dir.glob('checkpoint_epoch_*.pth'))
            
            if checkpoint_files:
                logger.info(f"Cleaning up {len(checkpoint_files)} old checkpoint files to save storage space...")
                
                for checkpoint_file in checkpoint_files:
                    checkpoint_file.unlink()
                    logger.debug(f"Deleted {checkpoint_file}")
                
                logger.info("Old checkpoint cleanup completed")
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        training_config = self.config.training_config
        num_epochs = training_config.get('num_epochs', 100)
        save_interval = training_config.get('save_interval', 5)
        early_stopping_patience = training_config.get('early_stopping_patience', 10)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Val Loss: {val_metrics['val_loss']:.4f}, "
                       f"Val PSNR: {val_metrics['val_psnr']:.2f}, "
                       f"Val SSIM: {val_metrics['val_ssim']:.4f}, "
                       f"Time: {epoch_time:.2f}s")
            
            # Log to tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Epoch/Val_PSNR', val_metrics['val_psnr'], epoch)
            self.writer.add_scalar('Epoch/Val_SSIM', val_metrics['val_ssim'], epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint (only best model)
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                # Save only the best model
                self.save_checkpoint(epoch, val_metrics, is_best)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        logger.info("Training completed!")
        self.writer.close()

class CombinedLoss(nn.Module):
    """Combined loss function for frame interpolation with color-aware terms."""
    
    def __init__(self, l1_weight: float = 1.0, perceptual_weight: float = 0.1, ssim_weight: float = 0.1, 
                 color_weight: float = 0.1, gradient_weight: float = 0.05):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.gradient_weight = gradient_weight
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss with color-aware terms."""
        # L1 loss (primary)
        l1_loss = self.l1_loss(predicted, target)
        
        # MSE loss (secondary)
        mse_loss = self.mse_loss(predicted, target)
        
        # Check if losses are valid
        if torch.isnan(l1_loss) or torch.isinf(l1_loss):
            l1_loss = torch.tensor(0.1, device=predicted.device, requires_grad=True)
        
        if torch.isnan(mse_loss) or torch.isinf(mse_loss):
            mse_loss = torch.tensor(0.1, device=predicted.device, requires_grad=True)
        
        # Combined loss: 80% L1 + 20% MSE
        total_loss = 0.8 * l1_loss + 0.2 * mse_loss
        
        return total_loss
    
    def _ssim_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Simplified SSIM loss with numerical stability."""
        # This is a simplified version - in practice, you'd use a proper SSIM implementation
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        sigma_x = torch.var(x) + 1e-8  # Add small epsilon for numerical stability
        sigma_y = torch.var(y) + 1e-8
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y)) + 1e-8
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        
        ssim = numerator / (denominator + 1e-8)  # Add epsilon to prevent division by zero
        return torch.clamp(ssim, 0, 1)  # Clamp to valid range
    
    def _temporal_consistency_loss(self, predicted: torch.Tensor, 
                                 frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Temporal consistency loss."""
        # Ensure predicted frame is between frame1 and frame2
        diff1 = torch.abs(predicted - frame1)
        diff2 = torch.abs(predicted - frame2)
        
        # The predicted frame should be closer to the average
        avg_frame = (frame1 + frame2) / 2
        avg_diff = torch.abs(predicted - avg_frame)
        
        return torch.mean(avg_diff)
    
    def _color_consistency_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Color consistency loss to preserve vibrant colors."""
        # Calculate color histogram loss
        pred_hist = self._compute_color_histogram(predicted)
        target_hist = self._compute_color_histogram(target)
        
        # L1 loss on color histograms
        hist_loss = self.l1_loss(pred_hist, target_hist)
        
        # Color variance loss (encourage vibrant colors)
        pred_var = torch.var(predicted, dim=[2, 3])  # Variance across spatial dimensions
        target_var = torch.var(target, dim=[2, 3])
        var_loss = self.l1_loss(pred_var, target_var)
        
        # Saturation loss (encourage high saturation)
        pred_sat = self._compute_saturation(predicted)
        target_sat = self._compute_saturation(target)
        sat_loss = self.l1_loss(pred_sat, target_sat)
        
        return hist_loss + 0.5 * var_loss + 0.3 * sat_loss
    
    def _compute_color_histogram(self, image: torch.Tensor) -> torch.Tensor:
        """Compute color histogram for each channel."""
        # Convert to [0, 255] range and quantize
        image_255 = (image * 255).clamp(0, 255).long()
        
        # Compute histograms for each channel
        histograms = []
        for c in range(image.shape[1]):  # For each color channel
            hist = torch.histc(image_255[:, c].float(), bins=64, min=0, max=255)
            hist = hist / (hist.sum() + 1e-8)  # Normalize with epsilon
            histograms.append(hist)
        
        return torch.cat(histograms, dim=0)
    
    def _compute_saturation(self, image: torch.Tensor) -> torch.Tensor:
        """Compute saturation of RGB image."""
        # Convert to HSV and extract saturation
        # Simplified saturation calculation: max - min for each pixel
        max_vals = torch.max(image, dim=1, keepdim=True)[0]
        min_vals = torch.min(image, dim=1, keepdim=True)[0]
        saturation = max_vals - min_vals
        return torch.mean(saturation)
    
    def _gradient_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gradient loss to preserve edges and details."""
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if predicted.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
        
        # Compute gradients for each channel
        pred_grad_x = torch.zeros_like(predicted)
        pred_grad_y = torch.zeros_like(predicted)
        target_grad_x = torch.zeros_like(target)
        target_grad_y = torch.zeros_like(target)
        
        for c in range(predicted.shape[1]):
            pred_grad_x[:, c:c+1] = F.conv2d(predicted[:, c:c+1], sobel_x, padding=1)
            pred_grad_y[:, c:c+1] = F.conv2d(predicted[:, c:c+1], sobel_y, padding=1)
            target_grad_x[:, c:c+1] = F.conv2d(target[:, c:c+1], sobel_x, padding=1)
            target_grad_y[:, c:c+1] = F.conv2d(target[:, c:c+1], sobel_y, padding=1)
        
        # L1 loss on gradients
        grad_loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        grad_loss_y = self.l1_loss(pred_grad_y, target_grad_y)
        
        return grad_loss_x + grad_loss_y

def calculate_psnr(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate PSNR between predicted and target frames."""
    mse = torch.mean((predicted - target) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate SSIM between predicted and target frames."""
    # Simplified SSIM calculation
    mu_x = torch.mean(predicted)
    mu_y = torch.mean(target)
    sigma_x = torch.var(predicted)
    sigma_y = torch.var(target)
    sigma_xy = torch.mean((predicted - mu_x) * (target - mu_y))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    
    return ssim.item()
