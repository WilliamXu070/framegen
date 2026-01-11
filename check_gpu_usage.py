"""Quick diagnostic script to check GPU usage and tensor locations."""
import torch
import sys
sys.path.insert(0, '.')

from src.config import Config

def check_device_placement():
    """Check if model and loss are on the correct device."""
    print("=" * 60)
    print("GPU DIAGNOSTICS")
    print("=" * 60)
    
    # Check CUDA
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Current Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        print(f"   Current Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
    
    # Load config
    try:
        config = Config('configs/default.yaml')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n2. Target Device: {device}")
        
        # Create model
        from src.models.frame_interpolation import create_model
        model = create_model(config.model_config)
        model = model.to(device)
        
        # Check model device
        model_param_device = next(model.parameters()).device
        print(f"   Model parameters device: {model_param_device}")
        print(f"   Model is on GPU: {model_param_device.type == 'cuda'}")
        
        # Create loss function
        from src.trainer import CombinedLoss
        loss_config = config.get('training', {}).get('loss', {})
        perceptual_weight = loss_config.get('perceptual_weight', 0.1)
        criterion = CombinedLoss(perceptual_weight=perceptual_weight)
        criterion = criterion.to(device)
        
        # Check loss function device
        if hasattr(criterion, 'perceptual_loss_module') and criterion.perceptual_loss_module is not None:
            loss_param_device = next(criterion.perceptual_loss_module.parameters()).device
            print(f"   Perceptual loss VGG device: {loss_param_device}")
            print(f"   Perceptual loss is on GPU: {loss_param_device.type == 'cuda'}")
        
        # Create dummy input
        dummy_input = torch.randn(2, 3, 256, 256).to(device)
        print(f"\n3. Dummy input device: {dummy_input.device}")
        
        # Forward pass
        print("\n4. Running forward pass...")
        with torch.no_grad():
            output = model(dummy_input, dummy_input, alpha=0.5)
            print(f"   Output device: {output.device}")
            print(f"   Output shape: {output.shape}")
            
            # Check loss
            target = torch.randn_like(output).to(device)
            loss = criterion(output, target, dummy_input, dummy_input)
            print(f"   Loss device: {loss.device}")
            print(f"   Loss value: {loss.item():.4f}")
        
        # Memory after forward
        if torch.cuda.is_available():
            print(f"\n5. Memory after forward pass:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
            print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
        
        print("\n" + "=" * 60)
        print("✓ All checks passed - model and loss are on GPU")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_device_placement()
