"""
Progress UI utilities for dataset download and model training.
"""

import time
import threading
import sys
import os
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from contextlib import contextmanager

@contextmanager
def suppress_logging():
    """Context manager to temporarily suppress logging output."""
    # Get all loggers
    loggers = [logging.getLogger(name) for name in logging.Logger.manager.loggerDict]
    
    # Store original levels
    original_levels = {logger: logger.level for logger in loggers}
    
    # Set all loggers to CRITICAL to suppress most output
    for logger in loggers:
        logger.setLevel(logging.CRITICAL)
    
    try:
        yield
    finally:
        # Restore original levels
        for logger, level in original_levels.items():
            logger.setLevel(level)

class ProgressTracker:
    """Base class for progress tracking with UI updates."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update_interval = 0.1  # Update UI every 100ms
        self._lock = threading.Lock()
        
    def update(self, increment: int = 1, extra_info: str = ""):
        """Update progress."""
        with self._lock:
            self.current += increment
            self.current = min(self.current, self.total)
            
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self._update_display(extra_info)
                self.last_update_time = current_time
    
    def set_progress(self, value: int, extra_info: str = ""):
        """Set absolute progress value."""
        with self._lock:
            self.current = min(value, self.total)
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self._update_display(extra_info)
                self.last_update_time = current_time
    
    def _update_display(self, extra_info: str = ""):
        """Update the display (to be overridden by subclasses)."""
        pass
    
    def finish(self, message: str = "Completed"):
        """Mark as finished."""
        with self._lock:
            self.current = self.total
            self._update_display(message)
            print()  # New line after progress bar

class DatasetDownloadProgress(ProgressTracker):
    """Progress tracker for dataset download with size and time estimates."""
    
    def __init__(self, total_size_gb: float, total_files: int):
        super().__init__(total_files, "Downloading UCF101 Dataset")
        self.total_size_gb = total_size_gb
        self.downloaded_size_gb = 0.0
        self.download_speed_mbps = 0.0
        self.eta_seconds = 0
        self.file_sizes = {}  # Track individual file sizes
        
    def update_file_size(self, file_size_bytes: int):
        """Update with actual file size."""
        self.file_sizes[self.current] = file_size_bytes
        
    def update(self, increment: int = 1, file_size_bytes: int = 0, extra_info: str = ""):
        """Update progress with file size information."""
        if file_size_bytes > 0:
            self.file_sizes[self.current] = file_size_bytes
            self.downloaded_size_gb += file_size_bytes / (1024**3)
        
        super().update(increment, extra_info)
    
    def _update_display(self, extra_info: str = ""):
        """Update download progress display."""
        if self.total == 0:
            return
            
        # Calculate progress percentage
        progress_pct = (self.current / self.total) * 100
        
        # Calculate download speed
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.download_speed_mbps = (self.downloaded_size_gb * 1024) / elapsed_time
        
        # Calculate ETA
        if self.download_speed_mbps > 0:
            remaining_gb = self.total_size_gb - self.downloaded_size_gb
            self.eta_seconds = (remaining_gb * 1024) / self.download_speed_mbps
        
        # Create progress bar
        bar_length = 50
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Format time
        eta_str = self._format_time(self.eta_seconds) if self.eta_seconds > 0 else "Calculating..."
        elapsed_str = self._format_time(elapsed_time)
        
        # Format sizes
        downloaded_str = f"{self.downloaded_size_gb:.2f} GB"
        total_str = f"{self.total_size_gb:.2f} GB"
        speed_str = f"{self.download_speed_mbps:.2f} MB/s"
        
        # Create display string
        display = (f"\r{self.description}: [{bar}] {progress_pct:6.2f}% "
                  f"({self.current}/{self.total}) "
                  f"Size: {downloaded_str}/{total_str} "
                  f"Speed: {speed_str} "
                  f"ETA: {eta_str} "
                  f"Elapsed: {elapsed_str}")
        
        if extra_info:
            display += f" | {extra_info}"
        
        # Clear line and print - use a longer clear to ensure complete overwrite
        with suppress_logging():
            sys.stdout.write('\r' + ' ' * 200 + '\r')  # Clear line with more spaces
            sys.stdout.write(display)
            sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Format time in HH:MM:SS format."""
        if seconds < 0:
            return "00:00:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

class TrainingProgressTracker(ProgressTracker):
    """Progress tracker for model training with convergence metrics."""
    
    def __init__(self, total_epochs: int):
        super().__init__(total_epochs, "Training Model")
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.val_psnr = 0.0
        self.val_ssim = 0.0
        self.learning_rate = 0.0
        self.convergence_rate = 0.0
        self.best_val_loss = float('inf')
        self.epoch_times = []
        self.loss_history = []
        
    def update_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                    val_psnr: float, val_ssim: float, learning_rate: float,
                    epoch_time: float, extra_info: str = ""):
        """Update training progress for an epoch."""
        with self._lock:
            self.current_epoch = epoch
            self.train_loss = train_loss
            self.val_loss = val_loss
            self.val_psnr = val_psnr
            self.val_ssim = val_ssim
            self.learning_rate = learning_rate
            
            # Track epoch times
            self.epoch_times.append(epoch_time)
            if len(self.epoch_times) > 10:  # Keep only last 10 epochs
                self.epoch_times.pop(0)
            
            # Track loss history for convergence
            self.loss_history.append(val_loss)
            if len(self.loss_history) > 20:  # Keep last 20 epochs
                self.loss_history.pop(0)
            
            # Calculate convergence rate
            if len(self.loss_history) >= 5:
                recent_losses = self.loss_history[-5:]
                self.convergence_rate = self._calculate_convergence_rate(recent_losses)
            
            # Update best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            self._update_display(extra_info)
    
    def update_batch(self, batch: int, total_batches: int, batch_loss: float, extra_info: str = ""):
        """Update progress for current batch within epoch."""
        with self._lock:
            self.current_batch = batch
            self.total_batches = total_batches
            self.train_loss = batch_loss
            self._update_display(extra_info)
    
    def _calculate_convergence_rate(self, recent_losses: list) -> float:
        """Calculate convergence rate based on recent loss changes."""
        if len(recent_losses) < 2:
            return 0.0
        
        # Calculate average change in loss
        changes = [recent_losses[i] - recent_losses[i-1] for i in range(1, len(recent_losses))]
        avg_change = sum(changes) / len(changes)
        
        # Convert to percentage (negative means improving)
        if recent_losses[0] != 0:
            return (avg_change / recent_losses[0]) * 100
        return 0.0
    
    def _update_display(self, extra_info: str = ""):
        """Update training progress display."""
        if self.total == 0:
            return
        
        # Calculate progress percentage
        progress_pct = (self.current_epoch / self.total) * 100
        
        # Calculate time estimates
        elapsed_time = time.time() - self.start_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        
        if avg_epoch_time > 0:
            remaining_epochs = self.total - self.current_epoch
            eta_seconds = remaining_epochs * avg_epoch_time
        else:
            eta_seconds = 0
        
        # Create progress bar
        bar_length = 50
        filled_length = int(bar_length * self.current_epoch // self.total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Format time
        eta_str = self._format_time(eta_seconds) if eta_seconds > 0 else "Calculating..."
        elapsed_str = self._format_time(elapsed_time)
        
        # Format metrics
        train_loss_str = f"{self.train_loss:.4f}"
        val_loss_str = f"{self.val_loss:.4f}"
        psnr_str = f"{self.val_psnr:.2f}"
        ssim_str = f"{self.val_ssim:.4f}"
        lr_str = f"{self.learning_rate:.2e}"
        conv_str = f"{self.convergence_rate:+.2f}%"
        
        # Create display string
        display = (f"\r{self.description}: [{bar}] {progress_pct:6.2f}% "
                  f"Epoch {self.current_epoch}/{self.total} "
                  f"Train Loss: {train_loss_str} "
                  f"Val Loss: {val_loss_str} "
                  f"PSNR: {psnr_str} "
                  f"SSIM: {ssim_str} "
                  f"LR: {lr_str} "
                  f"Conv: {conv_str} "
                  f"ETA: {eta_str} "
                  f"Elapsed: {elapsed_str}")
        
        if extra_info:
            display += f" | {extra_info}"
        
        # Clear line and print - use a longer clear to ensure complete overwrite
        with suppress_logging():
            sys.stdout.write('\r' + ' ' * 200 + '\r')  # Clear line with more spaces
            sys.stdout.write(display)
            sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Format time in HH:MM:SS format."""
        if seconds < 0:
            return "00:00:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

class ProgressUI:
    """Main UI coordinator for progress tracking."""
    
    def __init__(self):
        self.current_tracker = None
        self.is_active = False
        
    def start_dataset_download(self, total_size_gb: float, total_files: int) -> DatasetDownloadProgress:
        """Start dataset download progress tracking."""
        self.current_tracker = DatasetDownloadProgress(total_size_gb, total_files)
        self.is_active = True
        print(f"\n🚀 Starting UCF101 Dataset Download")
        print(f"📊 Total Size: {total_size_gb:.2f} GB")
        print(f"📁 Total Files: {total_files:,}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 120)
        return self.current_tracker
    
    def start_training(self, total_epochs: int) -> TrainingProgressTracker:
        """Start training progress tracking."""
        self.current_tracker = TrainingProgressTracker(total_epochs)
        self.is_active = True
        print(f"\n🎯 Starting Model Training")
        print(f"📈 Total Epochs: {total_epochs}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 120)
        return self.current_tracker
    
    def finish_current(self, message: str = "Completed"):
        """Finish current progress tracking."""
        if self.current_tracker:
            self.current_tracker.finish(message)
            self.is_active = False
            print(f"✅ {message}")
            print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 120)
    
    def get_current_tracker(self):
        """Get current active tracker."""
        return self.current_tracker

# Global UI instance
progress_ui = ProgressUI()
