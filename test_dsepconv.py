"""
Test script to diagnose dsepconv/CuPy issues.
This will tell you if dsepconv is working or using the fallback.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

print("=" * 60)
print("Testing dsepconv/CuPy Status")
print("=" * 60)

# Test 1: Check if dsepconv can be imported
print("\n[1/4] Checking dsepconv import...")
try:
    try:
        from cupy_module import dsepconv
            print("   [OK] dsepconv imported from cupy_module")
    except ImportError:
        try:
            from LDMVFI.cupy_module import dsepconv
            print("   [OK] dsepconv imported from LDMVFI.cupy_module")
        except ImportError:
            dsepconv = None
            print("   [FAIL] dsepconv NOT available (will use fallback)")
except Exception as e:
    print(f"   ✗ Error importing dsepconv: {e}")
    dsepconv = None

# Test 2: Check CuPy status
print("\n[2/4] Checking CuPy installation...")
try:
    import cupy as cp
    print(f"   [OK] CuPy version: {cp.__version__}")
    print(f"   [OK] CuPy CUDA available: {cp.cuda.is_available()}")
    if cp.cuda.is_available():
        print(f"   [OK] CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    print("   [FAIL] CuPy not installed")
    cp = None
except Exception as e:
    print(f"   [FAIL] CuPy error: {e}")
    cp = None

# Test 3: Try to actually use dsepconv if available
print("\n[3/4] Testing dsepconv CUDA kernel...")
if dsepconv is not None and cp is not None:
    try:
        # Create dummy tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {device}")
        
        if device.type == 'cuda':
            # Small test tensors
            batch_size = 1
            channels = 3
            height = 32
            width = 32
            filter_size = 5
            
            tensor_input = torch.randn(batch_size, channels, height + 4, width + 4).cuda()
            tensor_vertical = torch.randn(batch_size, filter_size, height, width).cuda()
            tensor_horizontal = torch.randn(batch_size, filter_size, height, width).cuda()
            tensor_offset_x = torch.randn(batch_size, filter_size, height, width).cuda()
            tensor_offset_y = torch.randn(batch_size, filter_size, height, width).cuda()
            tensor_mask = torch.randn(batch_size, filter_size * filter_size, height, width).cuda()
            
            # Try to call dsepconv
            print("   Attempting dsepconv operation...")
            result = dsepconv.FunctionDSepconv(
                tensor_input, tensor_vertical, tensor_horizontal,
                tensor_offset_x, tensor_offset_y, tensor_mask
            )
            print(f"   [OK] dsepconv WORKS! Output shape: {result.shape}")
            print("   -> This means CuPy is properly compiled for your GPU")
            
        else:
            print("   [WARN] CUDA not available, cannot test dsepconv")
            
    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA_ERROR_NO_BINARY_FOR_GPU" in error_msg or "no kernel image" in error_msg:
            print("   [FAIL] dsepconv FAILS with CUDA compute capability error")
            print("   -> This confirms the CuPy recompilation issue!")
            print(f"   Error: {error_msg[:100]}...")
        else:
            print(f"   [FAIL] dsepconv FAILS with error: {error_msg[:100]}...")
    except Exception as e:
        print(f"   [FAIL] dsepconv FAILS with error: {type(e).__name__}: {str(e)[:100]}...")
else:
    print("   [WARN] Cannot test - dsepconv or CuPy not available")

# Test 4: Check what the decoder would use
print("\n[4/4] Testing decoder fallback code path...")
try:
    from src.models.modules.diffusionmodules.model import FlowDecoderWithResidual
    print("   [OK] Decoder imported successfully")
    
    # Check if decoder has the fallback warning flag
    print("\n   The decoder will use:")
    if dsepconv is None:
        print("   -> Fallback code (dsepconv not available)")
    else:
        print("   -> dsepconv CUDA kernel (if it works)")
        print("   -> Fallback code (if dsepconv fails at runtime)")
        
except ImportError as e:
    print(f"   [FAIL] Could not import decoder: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if dsepconv is None:
    print("[FAIL] PROBLEM CONFIRMED: dsepconv module not available")
    print("   -> Training will use slow fallback code")
    print("   -> This explains the extreme slowdown")
elif cp is None:
    print("[FAIL] PROBLEM CONFIRMED: CuPy not installed")
    print("   -> Cannot use dsepconv without CuPy")
else:
    # If we got here and dsepconv works, we would have printed success
    print("[WARN] Status unclear - check test results above")
    print("   If dsepconv test FAILED -> CuPy recompilation needed")
    print("   If dsepconv test PASSED -> Problem is elsewhere")

print("\nTo fix CuPy issue, see: RECOMPILE_CUPY_STEPS.md")
print("=" * 60)
