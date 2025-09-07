"""
Tests for model architectures.
"""

import torch
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.frame_interpolation import FrameInterpolationNet, AdvancedFrameInterpolationNet, create_model

class TestFrameInterpolationNet:
    """Test cases for FrameInterpolationNet."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = FrameInterpolationNet()
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = FrameInterpolationNet()
        batch_size = 2
        height, width = 256, 256
        channels = 3
        
        frame1 = torch.randn(batch_size, channels, height, width)
        frame2 = torch.randn(batch_size, channels, height, width)
        
        with torch.no_grad():
            output = model(frame1, frame2)
        
        assert output.shape == (batch_size, channels, height, width)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_interpolate_multiple_frames(self):
        """Test multiple frame interpolation."""
        model = FrameInterpolationNet()
        batch_size = 1
        height, width = 256, 256
        channels = 3
        
        frame1 = torch.randn(batch_size, channels, height, width)
        frame2 = torch.randn(batch_size, channels, height, width)
        num_frames = 3
        
        with torch.no_grad():
            frames = model.interpolate_multiple_frames(frame1, frame2, num_frames)
        
        assert len(frames) == num_frames
        for frame in frames:
            assert frame.shape == (batch_size, channels, height, width)

class TestAdvancedFrameInterpolationNet:
    """Test cases for AdvancedFrameInterpolationNet."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = AdvancedFrameInterpolationNet()
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = AdvancedFrameInterpolationNet()
        batch_size = 2
        height, width = 256, 256
        channels = 3
        
        frame1 = torch.randn(batch_size, channels, height, width)
        frame2 = torch.randn(batch_size, channels, height, width)
        
        with torch.no_grad():
            output = model(frame1, frame2)
        
        assert output.shape == (batch_size, channels, height, width)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output

class TestModelFactory:
    """Test cases for model factory function."""
    
    def test_create_optical_flow_model(self):
        """Test creating optical flow based model."""
        config = {
            'architecture': 'optical_flow_based',
            'input_channels': 3,
            'hidden_dim': 256,
            'num_layers': 4,
            'dropout': 0.1
        }
        
        model = create_model(config)
        assert model is not None
        assert isinstance(model, FrameInterpolationNet)
    
    def test_create_advanced_model(self):
        """Test creating advanced model."""
        config = {
            'architecture': 'advanced',
            'input_channels': 3,
            'hidden_dim': 256
        }
        
        model = create_model(config)
        assert model is not None
        assert isinstance(model, AdvancedFrameInterpolationNet)
    
    def test_invalid_architecture(self):
        """Test invalid architecture raises error."""
        config = {
            'architecture': 'invalid_architecture'
        }
        
        with pytest.raises(ValueError):
            create_model(config)

if __name__ == "__main__":
    pytest.main([__file__])
