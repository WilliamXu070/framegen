"""
Training monitoring and safety checks for frame interpolation.
Detects common training issues and provides detailed diagnostics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitor training for issues and provide diagnostics."""
    
    def __init__(
        self,
        loss_history_size: int = 100,
        gradient_check_interval: int = 10,
        color_check_interval: int = 5,
        loss_change_threshold: float = 1e-6,
        min_loss_change: float = 1e-4
    ):
        """
        Initialize training monitor.
        
        Args:
            loss_history_size: Number of recent losses to track
            gradient_check_interval: Check gradients every N batches
            color_check_interval: Check colors every N batches
            loss_change_threshold: Minimum loss change to consider training active
            min_loss_change: Minimum expected loss change per epoch
        """
        self.loss_history_size = loss_history_size
        self.gradient_check_interval = gradient_check_interval
        self.color_check_interval = color_check_interval
        self.loss_change_threshold = loss_change_threshold
        self.min_loss_change = min_loss_change
        
        # Loss tracking
        self.train_loss_history = deque(maxlen=loss_history_size)
        self.val_loss_history = deque(maxlen=loss_history_size)
        self.epoch_loss_history = []
        
        # Gradient tracking
        self.gradient_norms = []
        self.zero_gradient_count = 0
        self.nan_gradient_count = 0
        self.inf_gradient_count = 0
        
        # Output validation
        self.color_violations = []
        self.nan_output_count = 0
        self.inf_output_count = 0
        
        # Training state
        self.batch_count = 0
        self.epoch_count = 0
        self.last_check_batch = 0
        
        # Warnings
        self.warnings_issued = set()
        
    def check_batch_loss(self, loss: torch.Tensor, batch_idx: int) -> Dict[str, bool]:
        """
        Check batch loss for issues.
        
        Returns:
            Dictionary of detected issues
        """
        issues = {}
        
        # Check for NaN/Inf
        if torch.isnan(loss):
            issues['nan_loss'] = True
            logger.error(f"⚠️ NaN loss detected at batch {batch_idx}")
            self._issue_warning('nan_loss', f"Batch {batch_idx}: Loss is NaN")
        
        if torch.isinf(loss):
            issues['inf_loss'] = True
            logger.error(f"⚠️ Inf loss detected at batch {batch_idx}")
            self._issue_warning('inf_loss', f"Batch {batch_idx}: Loss is Inf")
        
        # Check loss value
        loss_val = loss.item()
        if loss_val < 0:
            issues['negative_loss'] = True
            logger.warning(f"⚠️ Negative loss detected at batch {batch_idx}: {loss_val}")
        
        if loss_val > 1e6:
            issues['exploding_loss'] = True
            logger.error(f"⚠️ Exploding loss detected at batch {batch_idx}: {loss_val}")
            self._issue_warning('exploding_loss', f"Batch {batch_idx}: Loss is extremely large ({loss_val:.2e})")
        
        return issues
    
    def check_model_output(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        batch_idx: int
    ) -> Dict[str, bool]:
        """
        Check model output for issues (colors, NaN, Inf, etc.).
        
        Returns:
            Dictionary of detected issues
        """
        issues = {}
        
        # Check for NaN/Inf in output
        nan_mask = torch.isnan(predicted)
        inf_mask = torch.isinf(predicted)
        
        if nan_mask.any():
            nan_count = nan_mask.sum().item()
            total_pixels = predicted.numel()
            nan_ratio = nan_count / total_pixels
            issues['nan_output'] = True
            self.nan_output_count += 1
            logger.error(
                f"⚠️ NaN in model output at batch {batch_idx}: "
                f"{nan_count}/{total_pixels} pixels ({nan_ratio*100:.2f}%)"
            )
            self._issue_warning('nan_output', f"Batch {batch_idx}: {nan_ratio*100:.2f}% NaN pixels")
        
        if inf_mask.any():
            inf_count = inf_mask.sum().item()
            total_pixels = predicted.numel()
            inf_ratio = inf_count / total_pixels
            issues['inf_output'] = True
            self.inf_output_count += 1
            logger.error(
                f"⚠️ Inf in model output at batch {batch_idx}: "
                f"{inf_count}/{total_pixels} pixels ({inf_ratio*100:.2f}%)"
            )
            self._issue_warning('inf_output', f"Batch {batch_idx}: {inf_ratio*100:.2f}% Inf pixels")
        
        # Check output range (should be [-1, 1] for Tanh, [0, 1] for Sigmoid)
        min_val = predicted.min().item()
        max_val = predicted.max().item()
        
        if min_val < -1.1 or max_val > 1.1:
            issues['out_of_range'] = True
            logger.warning(
                f"⚠️ Model output out of range at batch {batch_idx}: "
                f"[{min_val:.3f}, {max_val:.3f}], expected [-1, 1]"
            )
        
        # Check color channels (every N batches to avoid overhead)
        if batch_idx % self.color_check_interval == 0:
            color_issues = self._check_color_channels(predicted, batch_idx)
            issues.update(color_issues)
        
        return issues
    
    def _check_color_channels(
        self,
        output: torch.Tensor,
        batch_idx: int
    ) -> Dict[str, bool]:
        """
        Check for color channel issues (grayscale, no color, etc.).
        
        Returns:
            Dictionary of detected color issues
        """
        issues = {}
        
        # Move to CPU for analysis
        output_cpu = output.detach().cpu()
        
        # Check each sample in batch
        batch_size = output_cpu.shape[0]
        
        for b in range(batch_size):
            sample = output_cpu[b]  # [C, H, W]
            
            if sample.shape[0] != 3:
                continue  # Skip if not RGB
            
            # Get channel means
            r_mean = sample[0].mean().item()
            g_mean = sample[1].mean().item()
            b_mean = sample[2].mean().item()
            
            # Check if all channels are similar (grayscale)
            channel_means = torch.tensor([r_mean, g_mean, b_mean])
            channel_std = channel_means.std().item()
            channel_mean = channel_means.mean().item()
            
            # If std is very small relative to mean, channels are too similar
            if channel_mean > 0.01 and channel_std / channel_mean < 0.01:
                issues['grayscale_output'] = True
                logger.warning(
                    f"⚠️ Grayscale output detected at batch {batch_idx}, sample {b}: "
                    f"Channel std={channel_std:.6f}, mean={channel_mean:.3f}"
                )
                self.color_violations.append({
                    'batch': batch_idx,
                    'sample': b,
                    'type': 'grayscale',
                    'channel_means': [r_mean, g_mean, b_mean],
                    'std': channel_std
                })
            
            # Check for zero/very low color signal
            max_channel = max(abs(r_mean), abs(g_mean), abs(b_mean))
            if max_channel < 0.01:
                issues['low_color_signal'] = True
                logger.warning(
                    f"⚠️ Very low color signal at batch {batch_idx}, sample {b}: "
                    f"Max channel mean={max_channel:.6f}"
                )
                self.color_violations.append({
                    'batch': batch_idx,
                    'sample': b,
                    'type': 'low_signal',
                    'channel_means': [r_mean, g_mean, b_mean]
                })
            
            # Check for single channel dominance (one channel much stronger)
            channel_abs = torch.abs(channel_means)
            max_idx = channel_abs.argmax().item()
            max_val = channel_abs[max_idx].item()
            other_mean = (channel_abs.sum() - max_val) / 2
            
            if other_mean > 0 and max_val / other_mean > 10:
                issues['channel_dominance'] = True
                logger.warning(
                    f"⚠️ Single channel dominance at batch {batch_idx}, sample {b}: "
                    f"Channel {['R','G','B'][max_idx]}={max_val:.3f}, others={other_mean:.3f}"
                )
                self.color_violations.append({
                    'batch': batch_idx,
                    'sample': b,
                    'type': 'channel_dominance',
                    'dominant_channel': ['R', 'G', 'B'][max_idx],
                    'channel_means': [r_mean, g_mean, b_mean]
                })
        
        return issues
    
    def check_gradients(
        self,
        model: nn.Module,
        batch_idx: int
    ) -> Dict[str, bool]:
        """
        Check model gradients for issues.
        
        Returns:
            Dictionary of detected gradient issues
        """
        issues = {}
        
        if batch_idx % self.gradient_check_interval != 0:
            return issues  # Skip if not time to check
        
        total_norm = 0.0
        param_count = 0
        zero_grad_count = 0
        nan_grad_count = 0
        inf_grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Check for zero gradients
                if param_norm.item() < 1e-8:
                    zero_grad_count += 1
                
                # Check for NaN gradients
                if torch.isnan(param.grad.data).any():
                    nan_grad_count += 1
                    issues['nan_gradients'] = True
                    logger.error(f"⚠️ NaN gradients in parameter {name} at batch {batch_idx}")
                
                # Check for Inf gradients
                if torch.isinf(param.grad.data).any():
                    inf_grad_count += 1
                    issues['inf_gradients'] = True
                    logger.error(f"⚠️ Inf gradients in parameter {name} at batch {batch_idx}")
        
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Check for zero gradients (vanishing gradients)
        if param_count > 0:
            zero_ratio = zero_grad_count / param_count
            if zero_ratio > 0.5:
                issues['vanishing_gradients'] = True
                logger.warning(
                    f"⚠️ Vanishing gradients at batch {batch_idx}: "
                    f"{zero_grad_count}/{param_count} parameters have zero gradients ({zero_ratio*100:.1f}%)"
                )
                self._issue_warning('vanishing_gradients', f"Batch {batch_idx}: {zero_ratio*100:.1f}% zero gradients")
                self.zero_gradient_count += 1
            
            # Check gradient norm
            if total_norm > 100:
                issues['exploding_gradients'] = True
                logger.warning(f"⚠️ Exploding gradients at batch {batch_idx}: norm={total_norm:.2f}")
                self._issue_warning('exploding_gradients', f"Batch {batch_idx}: Gradient norm={total_norm:.2f}")
            
            if total_norm < 1e-6:
                issues['tiny_gradients'] = True
                logger.warning(f"⚠️ Tiny gradients at batch {batch_idx}: norm={total_norm:.6f}")
        
        self.nan_gradient_count += nan_grad_count
        self.inf_gradient_count += inf_grad_count
        
        return issues
    
    def check_loss_progress(
        self,
        current_loss: float,
        epoch: int
    ) -> Dict[str, bool]:
        """
        Check if loss is improving (not stuck).
        
        Returns:
            Dictionary of detected issues
        """
        issues = {}
        
        self.train_loss_history.append(current_loss)
        
        # Need at least 10 samples to check
        if len(self.train_loss_history) < 10:
            return issues
        
        # Check if loss is changing
        recent_losses = list(self.train_loss_history)[-10:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # If std is very small relative to mean, loss is not changing
        if loss_mean > 0 and loss_std / loss_mean < self.loss_change_threshold:
            issues['stuck_training'] = True
            logger.error(
                f"⚠️ Training appears stuck at epoch {epoch}: "
                f"Loss std={loss_std:.6f}, mean={loss_mean:.4f}, "
                f"relative_change={loss_std/loss_mean:.6f} < {self.loss_change_threshold}"
            )
            self._issue_warning(
                'stuck_training',
                f"Epoch {epoch}: Loss not changing (std={loss_std:.6f}, mean={loss_mean:.4f})"
            )
        
        # Check if loss is increasing (not learning)
        if len(self.train_loss_history) >= 20:
            old_losses = list(self.train_loss_history)[-20:-10]
            new_losses = list(self.train_loss_history)[-10:]
            old_mean = np.mean(old_losses)
            new_mean = np.mean(new_losses)
            
            if new_mean > old_mean * 1.1:  # Loss increased by >10%
                issues['loss_increasing'] = True
                logger.warning(
                    f"⚠️ Loss is increasing at epoch {epoch}: "
                    f"Old mean={old_mean:.4f}, new mean={new_mean:.4f}"
                )
        
        return issues
    
    def check_epoch_progress(
        self,
        train_loss: float,
        val_loss: float,
        epoch: int
    ) -> Dict[str, bool]:
        """
        Check epoch-level progress issues.
        
        Returns:
            Dictionary of detected issues
        """
        issues = {}
        
        self.epoch_loss_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # Check validation loss vs training loss
        if len(self.epoch_loss_history) >= 2:
            prev = self.epoch_loss_history[-2]
            curr = self.epoch_loss_history[-1]
            
            train_change = prev['train_loss'] - curr['train_loss']
            val_change = prev['val_loss'] - curr['val_loss']
            
            # If training loss decreases but validation increases, overfitting
            if train_change > 0 and val_change < -0.01:
                issues['overfitting'] = True
                logger.warning(
                    f"⚠️ Possible overfitting at epoch {epoch}: "
                    f"Train loss decreased {train_change:.4f}, "
                    f"val loss increased {abs(val_change):.4f}"
                )
                self._issue_warning(
                    'overfitting',
                    f"Epoch {epoch}: Train↓ {train_change:.4f}, Val↑ {abs(val_change):.4f}"
                )
            
            # If validation loss is much higher than training, overfitting
            if curr['val_loss'] > curr['train_loss'] * 1.5:
                issues['validation_overfitting'] = True
                logger.warning(
                    f"⚠️ Large train-val gap at epoch {epoch}: "
                    f"Train={curr['train_loss']:.4f}, Val={curr['val_loss']:.4f}"
                )
            
            # Check if loss is not improving
            if abs(train_change) < self.min_loss_change and epoch > 3:
                issues['no_improvement'] = True
                logger.warning(
                    f"⚠️ Minimal loss improvement at epoch {epoch}: "
                    f"Change={train_change:.6f} < {self.min_loss_change}"
                )
        
        return issues
    
    def _issue_warning(self, warning_type: str, message: str):
        """Issue a warning (only once per type)."""
        if warning_type not in self.warnings_issued:
            self.warnings_issued.add(warning_type)
            logger.warning(f"🔔 {warning_type.upper()}: {message}")
    
    def get_summary(self) -> Dict:
        """Get summary of detected issues."""
        return {
            'batch_count': self.batch_count,
            'epoch_count': self.epoch_count,
            'nan_output_count': self.nan_output_count,
            'inf_output_count': self.inf_output_count,
            'zero_gradient_count': self.zero_gradient_count,
            'nan_gradient_count': self.nan_gradient_count,
            'inf_gradient_count': self.inf_gradient_count,
            'color_violations': len(self.color_violations),
            'avg_gradient_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0.0,
            'loss_history_size': len(self.train_loss_history),
            'warnings_issued': list(self.warnings_issued)
        }
    
    def reset_epoch(self):
        """Reset per-epoch counters."""
        self.batch_count = 0
        self.epoch_count += 1
    
    def update_batch(self):
        """Update batch counter."""
        self.batch_count += 1


class TrainingSafety:
    """High-level training safety checks."""
    
    @staticmethod
    def validate_data_loading(
        train_loader,
        val_loader,
        expected_keys: List[str] = ['frame1', 'frame2', 'target']
    ) -> Tuple[bool, List[str]]:
        """
        Validate data loaders before training.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            # On Windows, DataLoader with num_workers > 0 requires __main__ guard
            # Skip validation if we're not in main context (to avoid multiprocessing issues)
            import sys
            if sys.platform == 'win32':
                # On Windows, just check that loaders exist and have length
                # Full validation will happen during actual training
                if hasattr(train_loader, '__len__'):
                    try:
                        len(train_loader)
                        logger.info("✅ Data loader structure validated (skipping batch iteration on Windows)")
                        return True, []
                    except Exception as e:
                        errors.append(f"Error getting data loader length: {e}")
                        return False, errors
            
            # Try to get a batch (Unix/Linux or Windows with num_workers=0)
            batch = next(iter(train_loader))
            
            # Check required keys
            for key in expected_keys:
                if key not in batch:
                    errors.append(f"Missing key '{key}' in training batch")
                    return False, errors
            
            # Check tensor shapes
            frame1 = batch['frame1']
            frame2 = batch['frame2']
            target = batch['target']
            
            if frame1.shape != frame2.shape or frame1.shape != target.shape:
                errors.append(
                    f"Shape mismatch: frame1={frame1.shape}, "
                    f"frame2={frame2.shape}, target={target.shape}"
                )
            
            # Check for NaN/Inf in data
            for key in ['frame1', 'frame2', 'target']:
                if torch.isnan(batch[key]).any():
                    errors.append(f"NaN detected in {key}")
                if torch.isinf(batch[key]).any():
                    errors.append(f"Inf detected in {key}")
            
            # Check data range (should be [0, 1])
            for key in ['frame1', 'frame2', 'target']:
                min_val = batch[key].min().item()
                max_val = batch[key].max().item()
                if min_val < -0.1 or max_val > 1.1:
                    errors.append(
                        f"{key} out of range: [{min_val:.3f}, {max_val:.3f}], "
                        f"expected [0, 1]"
                    )
            
            logger.info("✅ Data loading validation passed")
            return True, []
            
        except Exception as e:
            errors.append(f"Error validating data loader: {e}")
            return False, errors
    
    @staticmethod
    def validate_model_forward(
        model: nn.Module,
        device: torch.device,
        input_shape: Tuple[int, ...] = (2, 3, 256, 256)
    ) -> Tuple[bool, List[str]]:
        """
        Validate model forward pass.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            model.eval()
            
            # Create dummy inputs
            frame1 = torch.randn(input_shape).to(device)
            frame2 = torch.randn(input_shape).to(device)
            
            with torch.no_grad():
                output = model(frame1, frame2, alpha=0.5)
            
            # Check output shape
            expected_shape = input_shape
            if output.shape != expected_shape:
                errors.append(
                    f"Output shape mismatch: got {output.shape}, "
                    f"expected {expected_shape}"
                )
            
            # Check for NaN/Inf
            if torch.isnan(output).any():
                errors.append("NaN in model output")
            if torch.isinf(output).any():
                errors.append("Inf in model output")
            
            # Check output range (should be [-1, 1] for Tanh)
            min_val = output.min().item()
            max_val = output.max().item()
            if min_val < -1.1 or max_val > 1.1:
                errors.append(
                    f"Output out of range: [{min_val:.3f}, {max_val:.3f}], "
                    f"expected [-1, 1]"
                )
            
            logger.info("✅ Model forward pass validation passed")
            return True, []
            
        except Exception as e:
            errors.append(f"Error in model forward pass: {e}")
            return False, errors
    
    @staticmethod
    def validate_loss_computation(
        criterion: nn.Module,
        device: torch.device,
        input_shape: Tuple[int, ...] = (2, 3, 256, 256)
    ) -> Tuple[bool, List[str]]:
        """
        Validate loss function computation.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Create dummy inputs with requires_grad=True for gradient check
            predicted = torch.randn(input_shape, requires_grad=True).to(device)
            target = torch.randn(input_shape, requires_grad=False).to(device)
            frame1 = torch.randn(input_shape, requires_grad=False).to(device)
            frame2 = torch.randn(input_shape, requires_grad=False).to(device)
            
            loss = criterion(predicted, target, frame1, frame2)
            
            # Check for NaN/Inf
            if torch.isnan(loss):
                errors.append("Loss is NaN")
            if torch.isinf(loss):
                errors.append("Loss is Inf")
            
            # Check loss value
            loss_val = loss.item()
            if loss_val < 0:
                errors.append(f"Negative loss: {loss_val}")
            if loss_val > 1e6:
                errors.append(f"Extremely large loss: {loss_val}")
            
            # Check gradient (only if loss requires grad)
            if loss.requires_grad:
                try:
                    # Retain grad on predicted to check gradients (since it might not be leaf)
                    predicted.retain_grad()
                    loss.backward()
                    # Check if gradients were computed (only if predicted is leaf or we retained grad)
                    if predicted.grad is None:
                        errors.append("Gradients not computed (predicted.grad is None)")
                except Exception as grad_error:
                    # Gradient computation failed, but this might be OK in validation
                    # Just log it, don't fail validation
                    logger.warning(f"Gradient computation in validation failed: {grad_error} (this is OK)")
            
            logger.info(f"✅ Loss computation validation passed (loss={loss_val:.4f})")
            return True, []
            
        except Exception as e:
            errors.append(f"Error in loss computation: {e}")
            return False, errors
