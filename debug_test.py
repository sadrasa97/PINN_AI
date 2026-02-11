#!/usr/bin/env python
"""
Debug script to test the physics_informed module
"""
import torch
from physics_informed import ReactionDiffusionPDE

def test_laplacian():
    print("Creating PDE object...")
    pde = ReactionDiffusionPDE(spatial_resolution=64, dt=0.1)
    
    print("Testing with 4D tensor...")
    # Test with a normal 4D tensor
    test_tensor = torch.randn(2, 1, 16, 16)  # Smaller tensor for faster test
    print(f"Input shape: {test_tensor.shape}")
    result = pde.laplacian_2d(test_tensor)
    print(f"Output shape: {result.shape}")
    print("4D test passed!")
    
    print("\nTesting with 5D tensor...")
    # Test with a 5D tensor that might cause the original error
    test_tensor_5d = torch.randn(1, 2, 1, 16, 16)
    print(f"Input shape: {test_tensor_5d.shape}")
    try:
        result_5d = pde.laplacian_2d(test_tensor_5d)
        print(f"Output shape: {result_5d.shape}")
        print("5D tensor handled successfully!")
    except Exception as e:
        print(f"5D tensor failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_laplacian()