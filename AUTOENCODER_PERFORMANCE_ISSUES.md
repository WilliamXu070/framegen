# Autoencoder Training Performance Issues

## Critical Problems Identified

### 1. **Triple Encoding Per Forward Pass (MAJOR BOTTLENECK)**

**Location:** `src/models/autoencoder.py` lines 182-188

**Problem:**
- Each `forward()` call runs the encoder **3 times**:
  1. Line 184: `encode(input)` - encodes target frame
  2. Line 162: `encode(x_prev, ret_feature=True)` - encodes previous frame  
  3. Line 163: `encode(x_next, ret_feature=True)` - encodes next frame

**Impact:** The encoder is the most expensive part (6.8M parameters, convolutional network). Running it 3x per iteration means:
- 3x computation per batch
- 3x GPU memory bandwidth
- Massive slowdown

**This is BY DESIGN** (LDMVFI architecture requires encoding all 3 frames), but combined with other issues, it's extremely slow.

---

### 2. **dsepconv Fallback Code (CRITICAL)**

**Location:** `src/models/modules/diffusionmodules/model.py` lines 579-648

**Problem:**
- The `dsepconv` CUDA kernel fails (CUDA compute capability mismatch)
- Falls back to CPU-based operations:
  - Multiple `torch.nn.functional.interpolate()` calls (CPU operations)
  - Tensor operations that may run on CPU
  - No GPU acceleration

**Evidence:**
- GPU throttling (0-100%) indicates GPU is waiting for CPU
- Extremely slow training (CPU is bottleneck)

**Impact:**
- The fallback is 100-1000x slower than optimized CUDA kernel
- Causes GPU-CPU synchronization overhead
- Massive performance degradation

**Fix Required:** Recompile CuPy with RTX 5070 support OR disable the decoder's warping operations temporarily.

---

### 3. **Windows Multiprocessing Issues**

**Location:** `configs/ldmvfi/autoencoder.yaml` line 38

**Problem:**
- `num_workers: 8` on Windows can cause:
  - Data loader hangs
  - Process spawning issues
  - Memory overhead

**Impact:**
- Training may hang at sanity check
- Slower data loading
- But less critical than #1 and #2

---

### 4. **PyTorch CUDA Compatibility Warning**

**Problem:**
- PyTorch 2.7.1 + CUDA 11.8 doesn't support sm_120 (RTX 5070)
- Uses backward compatibility mode
- Reduced performance

**Impact:**
- Not as critical as #1 and #2, but contributes to slowdown

---

## Recommended Fixes (Priority Order)

### Fix #1: Optimize dsepconv Fallback (IMMEDIATE)

The fallback code should ensure all operations stay on GPU:

```python
# In src/models/modules/diffusionmodules/model.py line 620-628
# Ensure all tensors are on GPU before interpolation
if mask1_simple.shape[2] != padded_prev.shape[2] or mask1_simple.shape[3] != padded_prev.shape[3]:
    # Ensure device consistency
    device = padded_prev.device
    mask1_simple = mask1_simple.to(device)
    mask2_simple = mask2_simple.to(device)
    
    mask1_simple = torch.nn.functional.interpolate(
        mask1_simple, size=(padded_prev.shape[2], padded_prev.shape[3]),
        mode='bilinear', align_corners=False
    )
    mask2_simple = torch.nn.functional.interpolate(
        mask2_simple, size=(padded_next.shape[2], padded_next.shape[3]),
        mode='bilinear', align_corners=False
    )
```

### Fix #2: Reduce num_workers (IMMEDIATE)

Change `num_workers: 8` → `num_workers: 4` or `0` in config files.

### Fix #3: Recompile CuPy (IMPORTANT)

Follow `RECOMPILE_CUPY_STEPS.md` to recompile CuPy with RTX 5070 support.

### Fix #4: Upgrade PyTorch (OPTIONAL)

Upgrade to PyTorch 2.9+ with CUDA 13.0+ for native sm_120 support.

---

## Expected Performance After Fixes

- **Current:** ~1 iteration per 10+ minutes (not even 1 iteration completed)
- **After Fix #1 (GPU fallback):** ~5-10 iterations per minute
- **After Fix #3 (CuPy recompile):** ~20-50 iterations per minute
- **After Fix #4 (PyTorch upgrade):** ~50-100+ iterations per minute

The triple encoding (#1) is by design and cannot be easily fixed - it's part of the LDMVFI architecture. However, fixing the dsepconv fallback (#2) will provide the biggest immediate improvement.
