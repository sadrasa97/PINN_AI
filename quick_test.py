#!/usr/bin/env python
"""
Quick test to check the actual data dimensions and model expectations
"""
import torch
import pandas as pd
from config import get_default_config
from data_loader import get_dataloaders
from model import create_model

def test_dimensions():
    print("Testing data and model dimensions...")
    
    # Load config
    config = get_default_config()
    print(f"Config gene expression dim: {config.data.gene_expression_dim}")
    
    # Load data
    train_loader, _, _, scaler = get_dataloaders(config, use_real_data=True)
    batch = next(iter(train_loader))
    
    genomic_data = batch['genomic']
    print(f"Actual genomic data shape: {genomic_data.shape}")
    
    # Create model
    model = create_model(config)
    
    # Test forward pass with a small sample
    batch_size = 2
    images = torch.randn(batch_size, 3, 512, 512)
    genomic = torch.randn(batch_size, genomic_data.shape[1])  # Use actual genomic dim
    
    print(f"Trying forward pass with genomic shape: {genomic.shape}")
    
    try:
        output = model(images, genomic, return_physics=True)
        print("Forward pass successful!")
        print(f"Output keys: {list(output.keys())}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dimensions()