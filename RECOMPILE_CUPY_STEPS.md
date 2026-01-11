# How to Recompile CuPy for RTX 5070 (Windows PowerShell)

## Your GPU Info
- **GPU**: NVIDIA GeForce RTX 5070
- **Compute Capability**: 12.0 (sm_120)
- **Status**: Current PyTorch/CuPy binaries don't include sm_120 support

## Method 1: Automated Script (Recommended)

I've created a PowerShell script for you. Run it:

```powershell
.\recompile_cupy.ps1
```

## Method 2: Manual Steps

### Step 1: Uninstall existing CuPy (if installed)
```powershell
pip uninstall cupy -y
```

### Step 2: Set environment variable for RTX 5070
In PowerShell:
```powershell
$env:CUPY_NVCC_GENERATE_CODE = "arch=compute_120,code=sm_120"
```

### Step 3: Install CuPy from source
```powershell
pip install cupy --no-binary cupy
```

**Note**: This will take 10-30 minutes as it compiles from source.

### Step 4: Verify installation
```powershell
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}'); print(f'CUDA available: {cp.cuda.is_available()}')"
```

## Method 3: Install CuPy for specific CUDA version (if you know it)

If you know your CUDA version, you can try:
```powershell
# For CUDA 12.x
pip install cupy-cuda12x --no-binary cupy-cuda12x

# Then set the environment variable
$env:CUPY_NVCC_GENERATE_CODE = "arch=compute_120,code=sm_120"
```

## Important Notes

1. **CUDA Toolkit Required**: You need CUDA Toolkit installed with NVCC compiler. Check:
   ```powershell
   nvcc --version
   ```

2. **Build Time**: Compiling from source takes 10-30 minutes depending on your CPU.

3. **Persistent Environment Variable**: The `$env:CUPY_NVCC_GENERATE_CODE` variable only lasts for the current PowerShell session. To make it permanent:
   ```powershell
   [System.Environment]::SetEnvironmentVariable("CUPY_NVCC_GENERATE_CODE", "arch=compute_120,code=sm_120", "User")
   ```

4. **PyTorch Limitation**: Your PyTorch version also doesn't support sm_120 natively, but that's a separate issue. The CuPy recompile will fix the `dsepconv` module issue.

## Troubleshooting

### If compilation fails:
1. Check CUDA Toolkit is installed: `nvcc --version`
2. Ensure you have Visual Studio Build Tools installed (required for CUDA compilation on Windows)
3. Try installing a specific CuPy version:
   ```powershell
   pip install cupy==12.0.0 --no-binary cupy
   ```

### If you get "no module named cupy" after installation:
- Make sure you're using the same Python environment
- Check: `python -c "import sys; print(sys.executable)"`

## After Recompiling

Once CuPy is recompiled, the `dsepconv` module should work because:
1. CuPy will have sm_120 support
2. The `cupy_kernel()` in `LDMVFI/cupy_module/dsepconv.py` will automatically compile kernels for your GPU
3. The CUDA error should disappear

Test it by running your training again - the fallback warning should no longer appear!
