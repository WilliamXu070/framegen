# Frame Generation Application

A Python-based application for generating intermediate frames in low frame rate videos using advanced computer vision and machine learning techniques. This application can increase video frame rates by predicting and generating intermediate frames between existing ones.

## Features

- **Advanced Frame Interpolation**: Uses deep learning models to generate high-quality intermediate frames
- **Multiple Model Architectures**: Supports optical flow-based and advanced neural network architectures
- **Flexible Training Pipeline**: Complete training infrastructure with data augmentation, validation, and checkpointing
- **Real-time Processing**: Support for both batch processing and real-time frame interpolation
- **Post-processing**: Temporal smoothing and quality enhancement features
- **Configurable**: YAML-based configuration system for easy customization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd frame-generation-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data/{train,validation,test}
mkdir -p models logs output
```

## Quick Start

### Training a Model

1. Prepare your training data by placing video files in the `data/train` directory
2. Configure training parameters in `configs/default.yaml`
3. Start training:

```bash
python main.py --mode train --config configs/default.yaml
```

### Running Inference

1. Train a model or use a pre-trained model
2. Run inference on a video:

```bash
python main.py --mode inference --input path/to/input_video.mp4 --output path/to/output_video.mp4 --model models/best_model.pth
```

## Configuration

The application uses YAML configuration files. Key configuration sections:

### Model Configuration
```yaml
model:
  name: "FrameInterpolationNet"
  architecture: "optical_flow_based"  # or "advanced"
  input_channels: 3
  hidden_dim: 256
  num_layers: 4
  dropout: 0.1
```

### Training Configuration
```yaml
training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 100
  validation_split: 0.2
  early_stopping_patience: 10
  device: "cuda"  # or "cpu"
```

### Data Configuration
```yaml
data:
  input_fps: 15
  target_fps: 60
  frame_size: [256, 256]
  augmentation:
    enabled: true
    rotation_range: 10
    brightness_range: 0.2
    contrast_range: 0.2
    horizontal_flip: true
```

## Project Structure

```
frame-generation-app/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── configs/
│   └── default.yaml       # Default configuration
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── trainer.py         # Training pipeline
│   ├── inference.py       # Inference pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   └── frame_interpolation.py  # Model architectures
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py     # Dataset classes
│   └── utils/
│       ├── __init__.py
│       ├── logger.py      # Logging utilities
│       └── video_utils.py # Video processing utilities
├── data/                  # Data directories
│   ├── train/
│   ├── validation/
│   └── test/
├── models/                # Saved models
├── logs/                  # Training logs
└── output/                # Output videos
```

## Usage Examples

### Basic Training
```python
from src.config import Config
from src.trainer import FrameInterpolationTrainer

# Load configuration
config = Config('configs/default.yaml')

# Initialize trainer
trainer = FrameInterpolationTrainer(config)

# Start training
trainer.train()
```

### Basic Inference
```python
from src.config import Config
from src.inference import FrameInterpolationInference

# Load configuration
config = Config('configs/default.yaml')

# Initialize inference
inference = FrameInterpolationInference(config, 'models/best_model.pth')

# Process video
inference.process_video('input_video.mp4', 'output_video.mp4')
```

### Real-time Processing
```python
from src.inference import RealTimeFrameInterpolation

# Initialize real-time processor
rt_processor = RealTimeFrameInterpolation(config, 'models/best_model.pth')

# Process frames in real-time
for frame in video_stream:
    processed_frame = rt_processor.process_frame(frame)
    # Display or save processed frame
```

## Model Architectures

### FrameInterpolationNet
- Optical flow-based architecture
- Feature extraction with attention mechanism
- Frame synthesis network
- Suitable for most frame interpolation tasks

### AdvancedFrameInterpolationNet
- Multi-scale feature extraction
- Advanced optical flow estimation
- Skip connections for better detail preservation
- Better for complex motion patterns

## Training Tips

1. **Data Preparation**: Ensure your training videos have consistent quality and frame rates
2. **Batch Size**: Start with smaller batch sizes (4-8) and increase based on GPU memory
3. **Learning Rate**: Use learning rate scheduling for better convergence
4. **Data Augmentation**: Enable augmentation to improve model generalization
5. **Validation**: Monitor validation metrics to prevent overfitting

## Performance Optimization

1. **GPU Usage**: Use CUDA for faster training and inference
2. **Mixed Precision**: Enable mixed precision training for faster training
3. **Data Loading**: Use multiple workers for data loading
4. **Model Optimization**: Use model quantization for deployment

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or frame resolution
2. **Poor Quality Results**: Increase training epochs or adjust learning rate
3. **Slow Training**: Enable mixed precision or use more powerful GPU
4. **Video Loading Issues**: Check video format compatibility

### Logging

The application provides comprehensive logging:
- Training progress and metrics
- Model performance statistics
- Error messages and debugging information

Logs are saved in the `logs/` directory with timestamps.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for video processing capabilities
- PyTorch for deep learning framework
- TensorBoard for training visualization
- The computer vision research community for frame interpolation techniques
# framegen
# framegen
# framegen
