"""
Training pipeline for frame interpolation models.
"""

import torch
import torch.nn as nn
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
from src.utils.logger import setup_logger
from src.config import Config

logger = logging.getLogger(__name__)

class FrameInterpolationTrainer:
    """Trainer class for frame interpolation models."""
    
    def __init__(self, config: Config):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = self._setup_device()
        
        # Initialize model
        self.model = create_model(config.model_config)
        self.model.to(self.device)
        
        # RTX 5070 model compilation for better performance
        if config.get('hardware.compile_model', False) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
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
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_data_loaders(self):
        """Setup data loaders with UCF101 dataset support."""
        ucf101_config = self.config.get('ucf101', {})
        
        if ucf101_config.get('enabled', False):
            logger.info("Setting up UCF101 dataset...")
            
            # Initialize UCF101 dataset manager
            self.ucf101_manager = UCF101DatasetManager(self.config.config)
            
            # Prepare UCF101 dataset
            if not self.ucf101_manager.prepare_dataset():
                logger.error("Failed to prepare UCF101 dataset, falling back to standard dataset")
                self.train_loader, self.val_loader = create_data_loaders(self.config.config)
                return
            
            # Validate dataset
            if not self.ucf101_manager.validate_dataset():
                logger.error("UCF101 dataset validation failed, falling back to standard dataset")
                self.train_loader, self.val_loader = create_data_loaders(self.config.config)
                return
            
            # Get dataset info
            dataset_info = self.ucf101_manager.get_dataset_info()
            logger.info(f"UCF101 dataset info: {dataset_info}")
            
            # Create UCF101 data loaders
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
            patience=5,
            verbose=True
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
                with torch.cuda.amp.autocast():
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
                if use_amp:
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'Device': str(self.device)
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
        
        # Save regular checkpoint
        models_dir = Path(self.config.get('paths.models_dir', 'models'))
        models_dir.mkdir(exist_ok=True)
        
        checkpoint_path = models_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = models_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {metrics['val_loss']:.4f}")
    
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
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if epoch % save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        logger.info("Training completed!")
        self.writer.close()

class CombinedLoss(nn.Module):
    """Combined loss function for frame interpolation."""
    
    def __init__(self, l1_weight: float = 1.0, perceptual_weight: float = 0.1, ssim_weight: float = 0.1):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss."""
        # L1 loss
        l1_loss = self.l1_loss(predicted, target)
        
        # MSE loss
        mse_loss = self.mse_loss(predicted, target)
        
        # Perceptual loss (simplified - using MSE on features)
        perceptual_loss = mse_loss  # Simplified version
        
        # SSIM loss (simplified)
        ssim_loss = 1 - self._ssim_loss(predicted, target)
        
        # Temporal consistency loss
        temporal_loss = self._temporal_consistency_loss(predicted, frame1, frame2)
        
        # Combined loss
        total_loss = (self.l1_weight * l1_loss + 
                     self.perceptual_weight * perceptual_loss + 
                     self.ssim_weight * ssim_loss + 
                     0.1 * temporal_loss)
        
        return total_loss
    
    def _ssim_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Simplified SSIM loss."""
        # This is a simplified version - in practice, you'd use a proper SSIM implementation
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        sigma_x = torch.var(x)
        sigma_y = torch.var(y)
        sigma_xy = torch.mean((x - mu_x) * (y - mu_y))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        
        return ssim
    
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
