# Instructions to Fix CUDA Compatibility Issue for RTX 5070

## Problem
The `cupy_module` CUDA extension was compiled for older GPU architectures and doesn't support the RTX 5070's compute capability (12.0 / sm_120), causing:
```
CUDA_ERROR_NO_BINARY_FOR_GPU: no kernel image is available for execution on the device
```

## Solution 1: Recompile CuPy with RTX 5070 Support

The RTX 5070 uses **compute capability 12.0 (sm_120)**. You need to recompile CuPy to support this architecture.

### Steps:

1. **Uninstall existing CuPy:**
   ```bash
   pip uninstall cupy
   ```

2. **Install CuPy from source with compute capability 12.0:**
   ```bash
   # Set environment variable for your GPU's compute capability
   $env:CUPY_NVCC_GENERATE_CODE = "arch=compute_120,code=sm_120"
   
   # Build and install from source
   pip install cupy --no-binary cupy
   ```

3. **Verify installation:**
   ```python
   import cupy as cp
   print(f"CuPy version: {cp.__version__}")
   print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
   ```

## Solution 2: Recompile LDMVFI cupy_module

The `LDMVFI/cupy_module/dsepconv.py` uses CuPy's RawKernel which needs to be compiled for your GPU.

### Steps:

1. **Clear CuPy's cache:**
   ```bash
   # Clear CuPy's CUDA kernel cache
   python -c "import cupy; cupy.clear_memo()"
   ```

2. **The cupy_module should automatically recompile** when you run the code again, as it uses `cupy_kernel()` which compiles on-the-fly. However, you may need to:

   - Ensure your CUDA toolkit supports compute capability 12.0
   - Set environment variables:
     ```bash
     $env:CUPY_NVCC_GENERATE_CODE = "arch=compute_120,code=sm_120"
     ```

3. **Check CUDA toolkit compatibility:**
   - RTX 5070 requires CUDA Toolkit 12.0 or later
   - Verify your CUDA version:
     ```bash
     nvcc --version
     ```

## Solution 3: Use Fallback (Current Implementation)

The code now includes a fallback that works without the optimized CUDA kernel, though with reduced performance and quality. This allows training to proceed while you work on recompiling.

## Additional Notes

- **For Windows PowerShell** (your system), use `$env:` syntax instead of `export`
- **CUDA Toolkit**: Ensure you have CUDA 12.0+ installed for RTX 5070 support
- **Driver**: Update to latest NVIDIA drivers that support RTX 5070

## Quick Test

After recompiling, test with:
```python
import torch
import cupy as cp

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"CuPy CUDA available: {cp.cuda.is_available()}")
```

If all checks pass, the dsepconv module should work correctly.
