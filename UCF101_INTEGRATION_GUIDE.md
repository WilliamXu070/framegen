# 🎬 UCF101 Dataset Integration Guide

## Overview

This guide explains how to use the **UCF101 Action Recognition Dataset** for training your frame interpolation model. The system automatically downloads, processes, and segments the UCF101 dataset from Hugging Face.

## 🚀 **Automatic UCF101 Integration**

The system now includes **automatic UCF101 dataset integration** with the following features:

### ✅ **Automatic Dataset Management:**
- **Download Detection**: Checks if UCF101 is already downloaded
- **Hugging Face Integration**: Downloads from Hugging Face datasets
- **Fallback Download**: Alternative download if Hugging Face fails
- **Automatic Processing**: Converts videos to frame pairs for training
- **Smart Segmentation**: Automatically splits into train/val/test (70%/15%/15%)

### ✅ **Quality Assurance:**
- **Dataset Validation**: Comprehensive quality checks
- **Frame Quality Analysis**: Sharpness, brightness, contrast metrics
- **Motion Analysis**: Detects motion between frames
- **Stratified Splitting**: Maintains class distribution across splits

## 🛠️ **Quick Setup**

### **Method 1: Automatic Setup (Recommended)**
```cmd
# Run the UCF101 setup script
python scripts/setup_ucf101.py
```

### **Method 2: Training with Auto-Setup**
```cmd
# Training will automatically download and setup UCF101
python examples/train_ucf101_example.py
```

### **Method 3: Windows Batch File**
```cmd
# Double-click or run:
train_ucf101_rtx5070.bat
```

## ⚙️ **Configuration**

The UCF101 integration is configured in your YAML config file:

```yaml
# UCF101 Dataset Configuration
ucf101:
  enabled: true  # Enable UCF101 dataset
  interpolation_factor: 4  # Frames to generate between existing frames
  min_frames_per_video: 10  # Minimum frames per video
  download_from_huggingface: true  # Download from Hugging Face
  alternative_download: true  # Enable alternative download if HF fails
  train_ratio: 0.7  # Training set ratio
  val_ratio: 0.15   # Validation set ratio
  test_ratio: 0.15  # Test set ratio
  stratified_split: true  # Use stratified splitting by class
  quality_check: true  # Enable quality checks
```

## 📊 **UCF101 Dataset Information**

### **Dataset Details:**
- **Total Videos**: ~13,320 action videos
- **Classes**: 101 action categories
- **Resolution**: Variable (automatically resized to your config)
- **Duration**: 1-10 seconds per video
- **Frames**: 10-300 frames per video

### **Action Categories Include:**
- Sports (basketball, soccer, tennis, etc.)
- Daily activities (brushing teeth, cooking, etc.)
- Human-object interactions
- Body movements and gestures

## 🔄 **Automatic Workflow**

### **1. Dataset Detection**
```python
# System automatically checks if UCF101 exists
if ucf101_manager.is_dataset_downloaded():
    logger.info("UCF101 dataset found, using existing data")
else:
    logger.info("UCF101 dataset not found, downloading...")
```

### **2. Download Process**
```python
# Downloads from Hugging Face
dataset = load_dataset("UCF101", split="train")

# Fallback to alternative source if needed
if not huggingface_success:
    download_ucf101_alternative()
```

### **3. Processing Pipeline**
```python
# Converts videos to frame pairs
for video in dataset:
    frames = extract_frames(video)
    frame_pairs = create_consecutive_pairs(frames)
    save_processed_video(frame_pairs)
```

### **4. Segmentation**
```python
# Stratified splitting by action class
train_videos, val_videos, test_videos = stratified_split(
    videos, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)
```

## 📈 **Training with UCF101**

### **Automatic Training Integration:**
The training pipeline automatically detects and uses UCF101:

```python
# In trainer.py
def _setup_data_loaders(self):
    ucf101_config = self.config.get('ucf101', {})
    
    if ucf101_config.get('enabled', False):
        # Setup UCF101 dataset
        self.ucf101_manager = UCF101DatasetManager(self.config.config)
        self.ucf101_manager.prepare_dataset()
        self.train_loader, self.val_loader = create_ucf101_data_loaders(self.config.config)
    else:
        # Use standard dataset
        self.train_loader, self.val_loader = create_data_loaders(self.config.config)
```

### **Training Commands:**

#### **RTX 5070 Optimized:**
```cmd
# Use the batch file
train_ucf101_rtx5070.bat

# Or command line
python examples/train_ucf101_example.py
```

#### **Standard Training:**
```cmd
python main.py --mode train --config configs/default.yaml
```

## 🔍 **Dataset Validation**

### **Automatic Quality Checks:**
- **Frame Quality**: Sharpness, brightness, contrast analysis
- **Motion Detection**: Analyzes motion between consecutive frames
- **Size Validation**: Ensures frames match target resolution
- **Count Validation**: Verifies minimum frames per video

### **Validation Report:**
```python
# Generates detailed validation report
validator = DatasetValidator(config.config)
validation_results = validator.validate_ucf101_dataset('data')
validator.generate_validation_report(validation_results, 'logs/ucf101_validation_report.json')
```

### **Report Contents:**
- Overall dataset validity
- Split-wise statistics
- Quality metrics per video
- Motion analysis results
- Recommendations for improvement

## 📁 **File Structure**

```
data/
├── ucf101/                    # UCF101 dataset directory
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Processed videos and metadata
├── train/                     # Training split
│   ├── video_000000/         # Video directories
│   │   ├── frame_000000.jpg  # Frame files
│   │   └── frame_000001.jpg
│   └── train_metadata.json   # Split metadata
├── validation/                # Validation split
└── test/                     # Test split
```

## 🎯 **Benefits of UCF101 Integration**

### **1. Large-Scale Training:**
- **13,320+ videos** for comprehensive training
- **Diverse action categories** for robust generalization
- **High-quality videos** with good motion patterns

### **2. Real-World Scenarios:**
- **Human actions** relevant to video interpolation
- **Varied lighting conditions** for robustness
- **Different camera angles** and movements

### **3. Automatic Management:**
- **No manual download** required
- **Automatic processing** and segmentation
- **Quality validation** built-in
- **Easy configuration** via YAML

## 🚨 **Troubleshooting**

### **Common Issues:**

#### **1. Download Failures:**
```cmd
# Check internet connection
# Verify Hugging Face access
# Try alternative download source
```

#### **2. Memory Issues:**
```yaml
# Reduce batch size in config
training:
  batch_size: 8  # Reduce from 24

# Or reduce frame size
data:
  frame_size: [256, 256]  # Reduce from [1024, 1024]
```

#### **3. Quality Issues:**
```cmd
# Check validation report
# Review logs/ucf101_validation_report.json
# Re-run dataset preparation if needed
```

### **Debug Commands:**
```cmd
# Check dataset status
python -c "from src.data.ucf101_dataset import UCF101DatasetManager; print(UCF101DatasetManager({}).is_dataset_downloaded())"

# Validate dataset
python scripts/setup_ucf101.py

# Check dataset info
python -c "from src.data.ucf101_dataset import UCF101DatasetManager; print(UCF101DatasetManager({}).get_dataset_info())"
```

## 📊 **Performance Expectations**

### **RTX 5070 Performance:**
- **Download Time**: 5-10 minutes (depending on internet)
- **Processing Time**: 10-15 minutes for full dataset
- **Training Time**: 2-4 hours for 100 epochs
- **Memory Usage**: 15-20GB VRAM for 1024x1024 resolution

### **Dataset Statistics:**
- **Total Frame Pairs**: ~200,000+ pairs
- **Training Pairs**: ~140,000 pairs
- **Validation Pairs**: ~30,000 pairs
- **Test Pairs**: ~30,000 pairs

## 🎉 **Ready to Train!**

Your frame interpolation model is now ready to train on the UCF101 Action Recognition Dataset! The system will automatically:

1. ✅ **Download** UCF101 from Hugging Face
2. ✅ **Process** videos into frame pairs
3. ✅ **Segment** into train/val/test splits
4. ✅ **Validate** dataset quality
5. ✅ **Train** your model with RTX 5070 optimizations

**Start training with:**
```cmd
train_ucf101_rtx5070.bat
```

**Happy training!** 🚀
