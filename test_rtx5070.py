"""
Quick test script to verify RTX 5070 works with PyTorch training.
"""
import torch
import torch.nn as nn
import time

print("="*60)
print("RTX 5070 Training Test")
print("="*60)

# Check GPU
print(f"\n1. GPU Detection:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")

# Simple model
print(f"\n2. Creating test model...")
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
).cuda()

print(f"   Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward pass
print(f"\n3. Testing forward pass...")
x = torch.randn(16, 128, device='cuda')
try:
    y = model(x)
    print(f"   ✓ Forward pass successful! Output shape: {y.shape}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    exit(1)

# Test backward pass
print(f"\n4. Testing backward pass (training simulation)...")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

try:
    target = torch.randn(16, 128, device='cuda')
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"   ✓ Backward pass successful! Loss: {loss.item():.4f}")
except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")
    exit(1)

# Benchmark training speed
print(f"\n5. Benchmarking training speed...")
model.train()
num_iterations = 100
batch_size = 32

start_time = time.time()
for i in range(num_iterations):
    x = torch.randn(batch_size, 128, device='cuda')
    target = torch.randn(batch_size, 128, device='cuda')

    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if (i + 1) % 25 == 0:
        print(f"   Iteration {i+1}/{num_iterations} - Loss: {loss.item():.4f}")

elapsed_time = time.time() - start_time
iterations_per_sec = num_iterations / elapsed_time

print(f"\n   ✓ Benchmark complete!")
print(f"   Total time: {elapsed_time:.2f}s")
print(f"   Speed: {iterations_per_sec:.1f} iterations/sec")
print(f"   GPU Memory Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print(f"\n" + "="*60)
print("✓ ALL TESTS PASSED - RTX 5070 IS WORKING!")
print("="*60)
print(f"\nThe sm_120 warnings you see are expected.")
print(f"PyTorch will use JIT compilation for unsupported kernels.")
print(f"Training will work, but may be slightly slower than future")
print(f"PyTorch versions with native sm_120 support.")
print("="*60)
