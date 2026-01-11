# RTX 5070 Migration Guide

## Problem Summary

Your RTX 5070 GPU uses the **Blackwell architecture** with **compute capability sm_120**, which is not supported by your current PyTorch installation (PyTorch 2.7.1 + CUDA 11.8).

**Error:**
```
NVIDIA GeForce RTX 5070 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90
```

## Root Causes

1. **PyTorch 2.7.1 + CUDA 11.8** does not support sm_120 (Blackwell architecture)
2. **pytorch-lightning 1.7.7** is incompatible with PyTorch 2.x
3. **CUDA 11.8** is too old for RTX 5070 - need CUDA 12.4+
4. Your project dependencies are from 2022 and need modernization

## Solutions (Choose One)

---

### **✅ Solution 1: Complete Environment Rebuild (RECOMMENDED)**

This creates a fresh environment with all compatible dependencies.

#### Step 1: Backup current environment
```bash
# Save current package list
pip freeze > old_requirements_backup.txt
conda env export > old_environment_backup.yaml
```

#### Step 2: Create new conda environment
```bash
# Remove old environment (optional)
conda deactivate
conda env remove -n ldmvfi  # or your current env name

# Create new environment with RTX 5070 support
conda env create -f LDMVFI/environment_rtx5070.yaml
conda activate ldmvfi_rtx5070
```

#### Step 3: Verify installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Expected output:**
```
PyTorch: 2.4.x (or newer)
CUDA: 12.6
CUDA Available: True
GPU: NVIDIA GeForce RTX 5070
```

---

### **✅ Solution 2: Pip-Only Upgrade (Faster, Less Safe)**

If conda is slow or you prefer pip, use this approach.

#### Step 1: Uninstall old PyTorch
```bash
pip uninstall torch torchvision torchaudio -y
```

#### Step 2: Install PyTorch with CUDA 12.6
```bash
pip install torch>=2.4.0 torchvision>=0.19.0 torchaudio>=2.4.0 --index-url https://download.pytorch.org/whl/cu126
```

#### Step 3: Upgrade PyTorch Lightning
```bash
pip install pytorch-lightning>=2.1.0 torchmetrics>=1.0.0 lightning-utilities>=0.10.0 --upgrade
```

#### Step 4: Update other dependencies
```bash
pip install -r requirements_rtx5070.txt
```

---

### **⚠️ Solution 3: Use PyTorch Nightly (Cutting Edge)**

If stable PyTorch still shows warnings, use nightly builds with latest CUDA support.

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch nightly with CUDA 12.6
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# Update PyTorch Lightning
pip install pytorch-lightning>=2.1.0 torchmetrics>=1.0.0 --upgrade
```

---

## Code Migration Issues

After updating dependencies, you'll need to fix compatibility issues:

### 1. **PyTorch Lightning 1.x → 2.x Changes**

**Old code (Lightning 1.x):**
```python
from pytorch_lightning import Trainer
trainer = Trainer(gpus=1)  # Deprecated
```

**New code (Lightning 2.x):**
```python
from pytorch_lightning import Trainer
trainer = Trainer(accelerator="gpu", devices=1)  # New API
```

### 2. **Trainer.add_argparse_args() Deprecated**

In `src/trainer_ldmvfi.py` (lines 68-73, 373-374, 530-531):

**Old code:**
```python
if hasattr(Trainer, "add_argparse_args"):
    parser = Trainer.add_argparse_args(parser)
```

**Fix:** Use `LightningCLI` or manually define arguments. For quick fix, wrap in try-except:
```python
try:
    parser = Trainer.add_argparse_args(parser)
except AttributeError:
    # Lightning 2.x doesn't have add_argparse_args
    pass
```

### 3. **checkpoint_callback Deprecated**

In `src/trainer_ldmvfi.py` (line 479-480):

**Old code:**
```python
if version.parse(pl.__version__) < version.parse('1.4.0'):
    trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)
```

**Fix:** Always use callbacks list in Lightning 2.x:
```python
# Remove the version check, always use callbacks
```

### 4. **CUDACallback Updates**

In `src/trainer_ldmvfi.py` (lines 345-362), update for Lightning 2.x:

**Old code:**
```python
torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
```

**New code:**
```python
# Check if using new strategy API
if hasattr(trainer.strategy, 'root_device'):
    device_idx = trainer.strategy.root_device.index
else:
    device_idx = 0  # Fallback for Lightning 2.x
torch.cuda.reset_peak_memory_stats(device_idx)
```

---

## Testing After Migration

### 1. **Test GPU Detection**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 2. **Test Training Pipeline**
```bash
# Test your main trainer
python src/trainer.py

# Test LDMVFI trainer
python src/trainer_ldmvfi.py -b configs/ldmvfi/your_config.yaml -t
```

### 3. **Check for Warnings**
Look for:
- ❌ Compute capability warnings → Need newer PyTorch
- ❌ `add_argparse_args` deprecation → Update code
- ❌ `gpus` parameter warnings → Use `accelerator="gpu", devices=1`

---

## Performance Optimizations for RTX 5070

Once everything works, enable these optimizations in `configs/default.yaml`:

```yaml
hardware:
  mixed_precision: true  # Enable FP16/BF16 for faster training
  compile_model: true  # Enable torch.compile() for 2x speedup
  cuda_benchmark: true  # Already enabled
  memory_efficient_attention: true  # Use Flash Attention if available
```

Also update batch size - RTX 5070 has 16GB VRAM (not 24GB as commented):
```yaml
training:
  batch_size: 8  # Adjust based on your model size
```

---

## Common Issues & Fixes

### Issue 1: "No module named 'pytorch_lightning.utilities.distributed'"
**Fix:** Update code to use `from lightning.pytorch.utilities.rank_zero import rank_zero_only`

### Issue 2: CUDA out of memory
**Fix:** Reduce batch size or enable gradient accumulation:
```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch size = 16
```

### Issue 3: "RuntimeError: Expected all tensors to be on the same device"
**Fix:** Ensure all model components are on CUDA:
```python
self.criterion.to(self.device)  # Already done in your trainer.py line 63
```

### Issue 4: Slow data loading
**Fix:** Increase num_workers (Windows specific):
```yaml
hardware:
  num_workers: 4  # Change from 0, test different values
  persistent_workers: true
```

---

## Quick Migration Checklist

- [ ] Backup current environment (`pip freeze`, `conda env export`)
- [ ] Install PyTorch 2.4+ with CUDA 12.6
- [ ] Upgrade pytorch-lightning to 2.1+
- [ ] Update all dependencies using `requirements_rtx5070.txt`
- [ ] Verify GPU detection (no sm_120 warnings)
- [ ] Fix Lightning 2.x API changes in code
- [ ] Test training pipeline with small batch
- [ ] Enable performance optimizations
- [ ] Run full training

---

## Support Resources

- PyTorch Installation: https://pytorch.org/get-started/locally/
- Lightning Migration Guide: https://lightning.ai/docs/pytorch/stable/upgrade/migration_guide.html
- NVIDIA CUDA 12.6 Download: https://developer.nvidia.com/cuda-12-6-0-download-archive
- RTX 5070 Compute Capability: https://developer.nvidia.com/cuda-gpus

---

## Need More Help?

If you encounter specific errors after following this guide, please provide:
1. Full error traceback
2. Output of `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
3. Output of `pip list | grep -E "torch|lightning"`
4. The specific command you're running
