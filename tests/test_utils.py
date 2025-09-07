"""
Tests for utility functions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.video_utils import VideoProcessor
from src.config import Config

class TestVideoProcessor:
    """Test cases for VideoProcessor."""
    
    def test_initialization(self):
        """Test VideoProcessor initialization."""
        processor = VideoProcessor()
        assert processor is not None
        assert processor.target_size == (256, 256)
    
    def test_custom_size_initialization(self):
        """Test VideoProcessor with custom size."""
        custom_size = (512, 512)
        processor = VideoProcessor(custom_size)
        assert processor.target_size == custom_size
    
    def test_preprocess_frame(self):
        """Test frame preprocessing."""
        processor = VideoProcessor()
        
        # Create test frame
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        processed = processor.preprocess_frame(frame)
        
        assert processed.shape == (256, 256, 3)
        assert processed.dtype == np.float32
        assert np.all(processed >= 0) and np.all(processed <= 1)
    
    def test_postprocess_frame(self):
        """Test frame postprocessing."""
        processor = VideoProcessor()
        
        # Create test frame (normalized)
        frame = np.random.rand(256, 256, 3).astype(np.float32)
        
        processed = processor.postprocess_frame(frame)
        
        assert processed.shape == (256, 256, 3)
        assert processed.dtype == np.uint8
        assert np.all(processed >= 0) and np.all(processed <= 255)
    
    def test_create_frame_pairs(self):
        """Test frame pair creation."""
        processor = VideoProcessor()
        
        # Create test frames
        frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(5)]
        
        pairs = processor.create_frame_pairs(frames)
        
        assert len(pairs) == 4  # 5 frames -> 4 pairs
        for pair in pairs:
            assert len(pair) == 2
            assert pair[0].shape == (256, 256, 3)
            assert pair[1].shape == (256, 256, 3)
    
    def test_interpolate_frames_linear(self):
        """Test linear frame interpolation."""
        processor = VideoProcessor()
        
        # Create test frames
        frame1 = np.zeros((256, 256, 3), dtype=np.uint8)
        frame2 = np.ones((256, 256, 3), dtype=np.uint8) * 255
        
        interpolated = processor.interpolate_frames_linear(frame1, frame2, 3)
        
        assert len(interpolated) == 3
        for frame in interpolated:
            assert frame.shape == (256, 256, 3)
            assert frame.dtype == np.uint8

class TestConfig:
    """Test cases for Config class."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Create a temporary config file
        import tempfile
        import yaml
        
        config_data = {
            'model': {
                'name': 'TestModel',
                'architecture': 'test'
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 0.01
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = Config(config_path)
            
            assert config.get('model.name') == 'TestModel'
            assert config.get('model.architecture') == 'test'
            assert config.get('training.batch_size') == 16
            assert config.get('training.learning_rate') == 0.01
            assert config.get('nonexistent.key', 'default') == 'default'
            
        finally:
            # Clean up
            Path(config_path).unlink()
    
    def test_config_setting(self):
        """Test configuration setting."""
        import tempfile
        import yaml
        
        config_data = {'test': {'value': 1}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = Config(config_path)
            
            config.set('test.new_value', 42)
            assert config.get('test.new_value') == 42
            
            config.set('new_section.key', 'value')
            assert config.get('new_section.key') == 'value'
            
        finally:
            # Clean up
            Path(config_path).unlink()

if __name__ == "__main__":
    pytest.main([__file__])
