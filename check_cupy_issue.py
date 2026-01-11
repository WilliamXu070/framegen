"""
Quick script to check if CuPy/dsepconv is the problem.
Run this to diagnose the slowdown issue.
"""

import torch
import sys
import os
import warnings

# Capture warnings
warnings.simplefilter('always')

print("=" * 60)
print("Checking if CuPy/dsepconv is causing slow training")
print("=" * 60)

# Test 1: Check CuPy
print("\n[1] Checking CuPy...")
try:
    import cupy as cp
    print(f"   CuPy version: {cp.__version__}")
    print(f"   CuPy CUDA available: {cp.cuda.is_available()}")
    cupy_ok = cp.cuda.is_available()
except Exception as e:
    print(f"   CuPy ERROR: {e}")
    cupy_ok = False

# Test 2: Try to import dsepconv
print("\n[2] Checking dsepconv module...")
dsepconv = None
try:
    from LDMVFI.cupy_module import dsepconv
    print("   dsepconv imported successfully")
except ImportError:
    print("   dsepconv NOT found (will use fallback)")
except Exception as e:
    print(f"   dsepconv import error: {e}")

# Test 3: Try to use dsepconv
print("\n[3] Testing dsepconv CUDA kernel...")
if dsepconv is not None and torch.cuda.is_available():
    try:
        device = torch.device('cuda')
        batch_size = 1
        channels = 3
        h, w = 32, 32
        filter_size = 5
        
        tensor_input = torch.randn(batch_size, channels, h+4, w+4).cuda()
        tensor_vertical = torch.randn(batch_size, filter_size, h, w).cuda()
        tensor_horizontal = torch.randn(batch_size, filter_size, h, w).cuda()
        tensor_offset_x = torch.randn(batch_size, filter_size, h, w).cuda()
        tensor_offset_y = torch.randn(batch_size, filter_size, h, w).cuda()
        tensor_mask = torch.randn(batch_size, filter_size*filter_size, h, w).cuda()
        
        result = dsepconv.FunctionDSepconv(
            tensor_input, tensor_vertical, tensor_horizontal,
            tensor_offset_x, tensor_offset_y, tensor_mask
        )
        print(f"   dsepconv WORKS! Output: {result.shape}")
        dsepconv_works = True
        
    except RuntimeError as e:
        error_str = str(e)
        if "CUDA_ERROR_NO_BINARY_FOR_GPU" in error_str or "no kernel image" in error_str:
            print("   dsepconv FAILS with CUDA compute capability error")
            print("   -> THIS IS THE PROBLEM! CuPy needs recompilation.")
            dsepconv_works = False
        else:
            print(f"   dsepconv FAILS: {error_str[:80]}...")
            dsepconv_works = False
    except Exception as e:
        print(f"   dsepconv FAILS: {type(e).__name__}: {str(e)[:80]}...")
        dsepconv_works = False
else:
    print("   Cannot test (dsepconv or CUDA not available)")
    dsepconv_works = False

# Test 4: Check training logs for warnings
print("\n[4] Summary...")
print("=" * 60)

if not cupy_ok:
    print("[PROBLEM] CuPy not working properly")
    print("   -> Fix: Install/reinstall CuPy")
elif not dsepconv_works:
    print("[PROBLEM CONFIRMED] dsepconv CUDA kernel fails")
    print("   -> This IS causing your slow training!")
    print("   -> Fix: Recompile CuPy (see RECOMPILE_CUPY_STEPS.md)")
    print("   -> Current: Using slow CPU fallback code")
    print("   -> Expected: 100-1000x speedup after fix")
else:
    print("[OK] dsepconv appears to work")
    print("   -> If training is still slow, problem is elsewhere")
    print("   -> Check: data loading, model size, batch size")

print("\nTo check during training, look for this warning:")
print('  "dsepconv CUDA kernel failed" or "Using simplified warping fallback"')
print("=" * 60)
