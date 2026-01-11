# num_workers: 0 vs > 0 on Windows - Performance Explained

## Important Clarification

**0 workers is NOT faster** - it's actually **slower** for data loading.

However, it **feels faster** because:
- `num_workers > 0` on Windows = **Hangs forever** (infinite time, no progress)
- `num_workers = 0` = **Works but slower** (finite time, actual progress)

So 0 is "faster" in terms of **getting results** vs. **hanging forever**.

---

## Performance Comparison

### num_workers = 4 (on Windows)
- **Data Loading Speed:** ✅ Fast (multi-threaded)
- **Training Progress:** ❌ **ZERO** (hangs forever)
- **Effective Speed:** **0 iterations/second** (hangs at sanity check)
- **Result:** Training never starts

### num_workers = 0 (on Windows)
- **Data Loading Speed:** ❌ Slower (single-threaded)
- **Training Progress:** ✅ **ACTUAL PROGRESS**
- **Effective Speed:** **0.1-0.5 iterations/second** (works!)
- **Result:** Training completes successfully

---

## Why It "Feels" Faster

**Scenario 1: num_workers = 4**
```
Time: 0 minutes  → Sanity Checking: 0it [00:00, ?it/s]
Time: 5 minutes  → Sanity Checking: 0it [00:00, ?it/s]  (still stuck)
Time: 15 minutes → Sanity Checking: 0it [00:00, ?it/s]  (still stuck)
Time: 1 hour     → Sanity Checking: 0it [00:00, ?it/s]  (still stuck)
Result: NEVER STARTS TRAINING
```

**Scenario 2: num_workers = 0**
```
Time: 0 minutes  → Sanity Checking: 0it [00:00, ?it/s]
Time: 30 seconds → Sanity Checking: 2/2 [00:30, 0.07it/s] ✓
Time: 1 minute   → Epoch 0: 1/6888 [00:10, 0.1it/s] ✓
Time: 2 minutes  → Epoch 0: 5/6888 [01:50, 0.05it/s] ✓
Result: TRAINING PROGRESSES!
```

**So 0 workers "feels faster" because:**
- ✅ Training actually starts
- ✅ You see progress
- ✅ You get results (even if slower)

vs.

- ❌ Training never starts
- ❌ No progress
- ❌ No results ever

---

## Data Loading Performance

### num_workers = 0 (single-threaded)
- **Speed:** Slower
- **CPU Usage:** 1 core (12.5% on 8-core CPU)
- **Memory:** Lower
- **Stability:** ✅ Works on Windows

### num_workers = 4 (multi-threaded)
- **Speed:** Faster (4x potential)
- **CPU Usage:** 4 cores (50% on 8-core CPU)
- **Memory:** Higher
- **Stability:** ❌ Hangs on Windows (spawn issues)

### num_workers = 8 (multi-threaded)
- **Speed:** Even faster (8x potential)
- **CPU Usage:** 8 cores (100% on 8-core CPU)
- **Memory:** Highest
- **Stability:** ❌ Hangs on Windows

---

## Why Windows Hangs with num_workers > 0

**Linux/Mac (fork model):**
- Workers created via `fork()` (fast, shares memory)
- ✅ Works perfectly
- ✅ num_workers = 8 is great

**Windows (spawn model):**
- Workers created via `spawn()` (slow, creates new processes)
- Each worker loads entire Python environment
- CUDA context initialization can deadlock
- File I/O can block
- ❌ Hangs at sanity check

---

## The Math

**num_workers = 4 (hanging):**
- Effective training speed: **0 iterations/hour** (hangs)
- Time to complete 1 epoch: **∞ (never)**

**num_workers = 0 (working):**
- Effective training speed: **360-1800 iterations/hour** (0.1-0.5 it/s)
- Time to complete 1 epoch (6414 iterations): **3.5-18 hours**

**Winner:** num_workers = 0 (even though slower, it actually works!)

---

## Bottom Line

**0 workers is NOT faster** - it's actually slower for data loading.

**BUT:**
- 0 workers = **Works** (slow but functional)
- > 0 workers = **Hangs** (fast but broken)

So we use 0 workers on Windows not because it's faster, but because **it's the only option that actually works**.

---

## If You Want Fast Data Loading on Windows

**Option 1: Use WSL2 (Windows Subsystem for Linux)**
- Run Linux inside Windows
- num_workers = 8 works perfectly
- Fast data loading
- Requires Linux environment setup

**Option 2: Use Linux**
- Native Linux or Linux VM
- num_workers = 8+ works great
- Best performance

**Option 3: Accept num_workers = 0**
- Works on Windows
- Slower data loading
- But training still works!

---

## Summary

- **0 workers:** Slower data loading, but **works** (training progresses)
- **> 0 workers:** Faster data loading, but **hangs** (training never starts)
- **On Windows:** Use 0 workers (only option that works)
- **On Linux:** Use 8+ workers (fast and stable)
