# Why Training Hangs at Sanity Check (Windows)

## The Problem

Your training is stuck at:
```
Sanity Checking: 0it [00:00, ?it/s]
```

For 15+ minutes. This is **NOT a CuPy problem** - it's a **Windows multiprocessing issue**.

---

## Root Cause: Windows DataLoader Multiprocessing

**Problem:**
- `num_workers: 4` on Windows causes DataLoader to hang
- Windows uses `spawn` instead of `fork` for multiprocessing
- This causes issues with:
  - Process spawning
  - Memory sharing
  - File handles
  - DLL loading

**Why it hangs at sanity check:**
- PyTorch Lightning runs a sanity check before training
- It tries to load validation batches
- DataLoader with `num_workers > 0` hangs waiting for worker processes
- The processes get stuck and never complete

---

## The Fix

**Set `num_workers: 0` on Windows:**

```yaml
data:
  params:
    num_workers: 0  # Must be 0 on Windows
```

This means:
- ✅ Training will work (no hanging)
- ⚠️ Data loading will be slower (single-threaded)
- ✅ Still much faster than hanging forever!

---

## Why This Happens

1. **Windows multiprocessing model:**
   - Linux/Mac: Uses `fork` (fast, shares memory)
   - Windows: Uses `spawn` (slow, creates new processes)
   - `spawn` can deadlock with CUDA/PyTorch

2. **PyTorch Lightning sanity check:**
   - Runs before training starts
   - Loads a few validation batches
   - If DataLoader hangs, sanity check never completes
   - Training never starts

3. **Why num_workers=4 still hangs:**
   - Even with fewer workers, Windows spawn model can hang
   - CUDA context initialization in worker processes can deadlock
   - File I/O in worker processes can block

---

## Solutions

### Solution 1: Set num_workers to 0 (RECOMMENDED)

**Pros:**
- Works immediately
- No hanging
- Simple fix

**Cons:**
- Slower data loading (single-threaded)
- But still faster than hanging!

**Apply:** Already fixed in config files

---

### Solution 2: Use persistent_workers (May still hang)

```yaml
num_workers: 4
persistent_workers: true
```

**Pros:**
- Faster data loading (if it works)
- Keeps workers alive between epochs

**Cons:**
- May still hang on Windows
- Uses more memory

**Not recommended on Windows**

---

### Solution 3: Use Linux/WSL2 (Best performance)

If you want fastest training:
- Use Linux or WSL2 (Windows Subsystem for Linux)
- `num_workers: 8` works perfectly
- Much faster data loading

---

## Expected Behavior After Fix

**Before (hanging):**
- Stuck at "Sanity Checking: 0it [00:00, ?it/s]"
- 15+ minutes with no progress
- Never starts training

**After (num_workers: 0):**
- Sanity check completes in seconds
- Training starts immediately
- Data loading is single-threaded but works

**Performance:**
- Data loading: Slower (but acceptable)
- Training speed: Same (GPU bound)
- Overall: Much better than hanging!

---

## Verification

After setting `num_workers: 0`:
1. Restart training
2. Sanity check should complete in < 30 seconds
3. Training should start immediately
4. You should see: `Epoch 0: 1/6888` within 1-2 minutes

If it still hangs after `num_workers: 0`, the problem is elsewhere (check data paths, file permissions, etc.).
