"""
Memory management utilities for preventing CUDA out of memory errors.
"""

import torch
import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MemoryManager:
    """Memory manager for CUDA memory optimization."""
    
    def __init__(self, device: torch.device, max_memory_usage: float = 0.85):
        """
        Initialize memory manager.
        
        Args:
            device: CUDA device
            max_memory_usage: Maximum memory usage threshold (0.0-1.0)
        """
        self.device = device
        self.max_memory_usage = max_memory_usage
        self.cleanup_interval = 10  # Clean up every N batches
        self.batch_count = 0
        
        if device.type == 'cuda':
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            self.max_allocated = int(self.total_memory * max_memory_usage)
            logger.info(f"Memory manager initialized: {self.total_memory / 1e9:.1f}GB total, "
                       f"max usage: {self.max_allocated / 1e9:.1f}GB")
    
    def check_memory_usage(self) -> float:
        """Check current memory usage as a fraction of total memory."""
        if self.device.type != 'cuda':
            return 0.0
        
        allocated = torch.cuda.memory_allocated(self.device)
        return allocated / self.total_memory
    
    def is_memory_high(self) -> bool:
        """Check if memory usage is above threshold."""
        return self.check_memory_usage() > self.max_memory_usage
    
    def cleanup_memory(self, force: bool = False):
        """Clean up CUDA memory."""
        if self.device.type != 'cuda':
            return
        
        if force or self.batch_count % self.cleanup_interval == 0:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Log memory status
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            logger.debug(f"Memory cleanup: allocated={allocated / 1e9:.2f}GB, "
                        f"reserved={reserved / 1e9:.2f}GB")
    
    def update_batch_count(self):
        """Update batch count for cleanup scheduling."""
        self.batch_count += 1
    
    def get_memory_info(self) -> dict:
        """Get detailed memory information."""
        if self.device.type != 'cuda':
            return {"device": "cpu", "memory_usage": 0.0}
        
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)
        
        return {
            "device": str(self.device),
            "total_memory_gb": self.total_memory / 1e9,
            "allocated_gb": allocated / 1e9,
            "reserved_gb": reserved / 1e9,
            "max_allocated_gb": max_allocated / 1e9,
            "usage_fraction": allocated / self.total_memory,
            "max_usage_fraction": max_allocated / self.total_memory
        }
    
    def log_memory_status(self):
        """Log current memory status."""
        info = self.get_memory_info()
        if self.device.type == 'cuda':
            logger.info(f"Memory status: {info['allocated_gb']:.2f}GB allocated, "
                       f"{info['reserved_gb']:.2f}GB reserved, "
                       f"usage: {info['usage_fraction']:.1%}")

def setup_memory_optimization(config: dict) -> MemoryManager:
    """Setup memory optimization based on configuration."""
    # Get device from training config
    training_config = config.get('training', {})
    device_str = training_config.get('device', 'auto')
    
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    # Get memory settings from hardware config
    hardware_config = config.get('hardware', {})
    max_memory_usage = hardware_config.get('memory_fraction', 0.8)
    
    # Set memory fraction if specified
    if device.type == 'cuda' and 'memory_fraction' in hardware_config:
        torch.cuda.set_per_process_memory_fraction(max_memory_usage)
        logger.info(f"Set CUDA memory fraction to {max_memory_usage}")
    
    # Create memory manager
    memory_manager = MemoryManager(device, max_memory_usage)
    
    # Enable memory growth if specified
    if hardware_config.get('allow_growth', False):
        # This is handled by PyTorch automatically
        logger.info("Memory growth enabled")
    
    return memory_manager

def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_memory_usage() -> dict:
    """Get current memory usage information."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(device),
        "total_memory_gb": total_memory / 1e9,
        "allocated_gb": allocated / 1e9,
        "reserved_gb": reserved / 1e9,
        "free_gb": (total_memory - allocated) / 1e9,
        "usage_percentage": (allocated / total_memory) * 100
    }
