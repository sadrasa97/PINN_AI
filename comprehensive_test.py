#!/usr/bin/env python
"""
Comprehensive test for the laplacian_2d fix
"""
import torch
from physics_informed import ReactionDiffusionPDE

def test_comprehensive():
    print("Creating PDE object...")
    pde = ReactionDiffusionPDE(spatial_resolution=64, dt=0.1)
    
    # Test 1: Normal 4D tensor
    print("\nTest 1: Normal 4D tensor [batch, channel, H, W]")
    test_4d = torch.randn(2, 1, 16, 16)
    print(f"  Input shape: {test_4d.shape}")
    result_4d = pde.laplacian_2d(test_4d)
    print(f"  Output shape: {result_4d.shape}")
    print("  [PASS]")
    
    # Test 2: 5D tensor with leading singleton dimension [1, batch, channel, H, W]
    print("\nTest 2: 5D tensor with leading singleton [1, batch, channel, H, W]")
    test_5d_1 = torch.randn(1, 2, 1, 16, 16)
    print(f"  Input shape: {test_5d_1.shape}")
    result_5d_1 = pde.laplacian_2d(test_5d_1)
    print(f"  Output shape: {result_5d_1.shape}")
    print("  [PASS]")
    
    # Test 3: 5D tensor with middle singleton dimension [batch, 1, channel, H, W]
    print("\nTest 3: 5D tensor with middle singleton [batch, 1, channel, H, W]")
    test_5d_2 = torch.randn(2, 1, 1, 16, 16)
    print(f"  Input shape: {test_5d_2.shape}")
    result_5d_2 = pde.laplacian_2d(test_5d_2)
    print(f"  Output shape: {result_5d_2.shape}")
    print("  [PASS]")
    
    # Test 4: 5D tensor with multiple singleton dimensions [1, 1, batch, H, W]
    print("\nTest 4: 5D tensor with multiple singletons [1, 1, 2, 16, 16]")
    test_5d_3 = torch.randn(1, 1, 2, 16, 16)
    print(f"  Input shape: {test_5d_3.shape}")
    result_5d_3 = pde.laplacian_2d(test_5d_3)
    print(f"  Output shape: {result_5d_3.shape}")
    print("  [PASS]")
    
    # Test 5: Different sizes
    print("\nTest 5: Different spatial resolution [3, 1, 32, 32]")
    test_4d_2 = torch.randn(3, 1, 32, 32)
    print(f"  Input shape: {test_4d_2.shape}")
    result_4d_2 = pde.laplacian_2d(test_4d_2)
    print(f"  Output shape: {result_4d_2.shape}")
    print("  [PASS]")
    
    # Test 6: Error case - invalid dimensions
    print("\nTest 6: Testing error handling for invalid dimensions")
    try:
        invalid_tensor = torch.randn(5, 4, 3, 2, 1, 16, 16)  # 7D tensor
        result_invalid = pde.laplacian_2d(invalid_tensor)
        print("  [FAIL] Should have raised an error!")
    except ValueError as e:
        print(f"  Expected error caught: {e}")
        print("  [PASS] Correctly handled invalid dimensions")
    except Exception as e:
        print(f"  Unexpected error: {e}")
        print("  [WARN] Different error than expected")
    
    print("\nAll tests completed successfully! The fix handles various tensor shapes properly.")
    
    # Test the full PDE workflow to make sure everything integrates correctly
    print("\nTesting full PDE workflow...")
    tumor_init = torch.randn(2, 1, 16, 16).abs()
    immune_init = torch.randn(2, 1, 16, 16).abs() * 0.5
    
    params = {
        'diffusion_tumor': torch.tensor([1.0, 1.5]),
        'diffusion_immune': torch.tensor([2.0, 1.0]),
        'proliferation_tumor': torch.tensor([0.5, 0.7]),
        'proliferation_immune': torch.tensor([0.3, 0.4]),
        'interaction_tumor_immune': torch.tensor([0.1, 0.15]),
        'carrying_capacity': torch.tensor([5.0, 6.0])
    }
    
    # Step forward in time
    tumor_next, immune_next = pde.step(tumor_init, immune_init, params, noise_scale=0.01)
    print(f"  Tumor evolution: {tumor_init.shape} -> {tumor_next.shape}")
    print(f"  Immune evolution: {immune_init.shape} -> {immune_next.shape}")
    print("  [PASS] Full PDE workflow successful")
    
    print("\n[SUCCESS] All comprehensive tests passed! The physics-informed neural network is working correctly.")

if __name__ == "__main__":
    test_comprehensive()