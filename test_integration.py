"""
Integration Test Suite for Physics-Informed Cancer Model
Tests all major components and their interactions
"""
import torch
import numpy as np
import sys
from pathlib import Path

print("="*80)
print("INTEGRATION TEST SUITE")
print("="*80)

def test_imports():
    """Test that all modules can be imported"""
    print("\n1. Testing imports...")
    try:
        from config import get_default_config
        from data_loader import TCGAMultimodalDataset, get_dataloaders, create_synthetic_tcga_data
        from model import create_model, PhysicsInformedCancerModel
        from physics_informed import ReactionDiffusionPDE, PhysicsInformedLoss, PhysicalParameterExtractor
        from losses import CoxProportionalHazardsLoss, VAELoss, CompletePINNLoss, compute_metrics
        from train import Trainer
        from evaluate import ModelEvaluator, create_analysis_report
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration system"""
    print("\n2. Testing configuration...")
    try:
        from config import get_default_config
        
        config = get_default_config()
        assert config.data.batch_size > 0
        assert config.training.learning_rate > 0
        assert config.model.image_output_dim > 0
        assert config.physics.spatial_resolution > 0
        
        print(f"✓ Configuration valid")
        print(f"  Device: {config.device}")
        print(f"  Batch size: {config.data.batch_size}")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False


def test_data_pipeline():
    """Test data loading pipeline"""
    print("\n3. Testing data pipeline...")
    try:
        from config import get_default_config
        from data_loader import get_dataloaders
        
        config = get_default_config()
        config.data.batch_size = 4
        
        train_loader, val_loader, test_loader, scaler = get_dataloaders(
            config, create_synthetic=True
        )
        
        # Test batch loading
        batch = next(iter(train_loader))
        
        assert 'image' in batch
        assert 'genomic' in batch
        assert 'survival_time' in batch
        assert batch['image'].shape[0] == config.data.batch_size
        assert batch['genomic'].shape[1] == config.data.gene_expression_dim
        
        print(f"✓ Data pipeline working")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Genomic shape: {batch['genomic'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_physics_components():
    """Test physics-informed components"""
    print("\n4. Testing physics components...")
    try:
        from physics_informed import ReactionDiffusionPDE, PhysicsInformedLoss
        
        # Test PDE solver
        pde = ReactionDiffusionPDE(spatial_resolution=32, dt=0.1)
        
        tumor_init = torch.randn(2, 1, 32, 32).abs()
        immune_init = torch.randn(2, 1, 32, 32).abs() * 0.5
        
        params = {
            'diffusion_tumor': torch.tensor([1.0, 2.0]),
            'diffusion_immune': torch.tensor([2.0, 1.5]),
            'proliferation_tumor': torch.tensor([0.5, 0.6]),
            'proliferation_immune': torch.tensor([0.3, 0.4]),
            'interaction_tumor_immune': torch.tensor([0.1, 0.15]),
            'carrying_capacity': torch.tensor([5.0, 4.5])
        }
        
        tumor_next, immune_next = pde.step(tumor_init, immune_init, params)
        
        assert tumor_next.shape == tumor_init.shape
        assert immune_next.shape == immune_init.shape
        
        # Test physics loss
        physics_loss = PhysicsInformedLoss(spatial_resolution=32)
        
        tumor_pred = torch.stack([tumor_init, tumor_next], dim=1)
        immune_pred = torch.stack([immune_init, immune_next], dim=1)
        
        loss, loss_dict = physics_loss(tumor_pred, immune_pred, params)
        
        assert loss.item() >= 0
        assert 'pde_residual' in loss_dict
        
        print(f"✓ Physics components working")
        print(f"  PDE residual: {loss_dict['pde_residual']:.6f}")
        print(f"  Boundary loss: {loss_dict['boundary']:.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Physics components failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_architecture():
    """Test complete model architecture"""
    print("\n5. Testing model architecture...")
    try:
        from config import get_default_config
        from model import create_model
        
        config = get_default_config()
        model = create_model(config)
        
        # Test forward pass
        batch_size = 2
        images = torch.randn(batch_size, 3, 512, 512)
        genomic = torch.randn(batch_size, config.data.gene_expression_dim)
        
        with torch.no_grad():
            output = model(images, genomic, return_physics=True)
        
        # Check outputs
        assert 'survival_risk' in output
        assert 'progression_logit' in output
        assert 'metastasis_logit' in output
        assert 'physics_params' in output
        assert 'initial_densities' in output
        assert 'attention_weights' in output
        
        assert output['survival_risk'].shape[0] == batch_size
        assert output['initial_densities'].shape[1] == 2  # tumor + immune
        
        print(f"✓ Model architecture working")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Output keys: {list(output.keys())[:5]}...")
        
        return True
    except Exception as e:
        print(f"✗ Model architecture failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_functions():
    """Test loss computation"""
    print("\n6. Testing loss functions...")
    try:
        from config import get_default_config
        from model import create_model
        from losses import CompletePINNLoss
        
        config = get_default_config()
        model = create_model(config)
        criterion = CompletePINNLoss(config)
        
        # Create dummy batch
        batch_size = 4
        images = torch.randn(batch_size, 3, 512, 512)
        genomic = torch.randn(batch_size, config.data.gene_expression_dim)
        
        targets = {
            'survival_time': torch.rand(batch_size) * 1000,
            'survival_event': torch.randint(0, 2, (batch_size,)).float(),
            'progression': torch.randint(0, 2, (batch_size,)),
            'metastasis': torch.randint(0, 2, (batch_size,)),
            'genomic': genomic
        }
        
        with torch.no_grad():
            output = model(images, genomic, return_physics=True)
            loss, loss_dict = criterion(output, targets, compute_physics=True)
        
        assert loss.item() >= 0
        assert 'total' in loss_dict
        assert 'clinical_total' in loss_dict
        assert 'vae_total' in loss_dict
        assert 'physics_total' in loss_dict
        
        print(f"✓ Loss functions working")
        print(f"  Total loss: {loss_dict['total']:.4f}")
        print(f"  Clinical: {loss_dict['clinical_total']:.4f}")
        print(f"  Physics: {loss_dict['physics_total']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test single training step"""
    print("\n7. Testing training step...")
    try:
        from config import get_default_config
        from model import create_model
        from losses import CompletePINNLoss
        
        config = get_default_config()
        model = create_model(config)
        criterion = CompletePINNLoss(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create dummy batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 512, 512)
        genomic = torch.randn(batch_size, config.data.gene_expression_dim)
        
        targets = {
            'survival_time': torch.rand(batch_size) * 1000,
            'survival_event': torch.randint(0, 2, (batch_size,)).float(),
            'progression': torch.randint(0, 2, (batch_size,)),
            'metastasis': torch.randint(0, 2, (batch_size,)),
            'genomic': genomic
        }
        
        # Forward pass
        output = model(images, genomic, return_physics=True)
        loss, loss_dict = criterion(output, targets, compute_physics=True)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients computed: {any(p.grad is not None for p in model.parameters())}")
        
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation utilities"""
    print("\n8. Testing evaluation...")
    try:
        from config import get_default_config
        from model import create_model
        from evaluate import ModelEvaluator
        
        config = get_default_config()
        model = create_model(config)
        evaluator = ModelEvaluator(model, config, device='cpu')
        
        # Test single patient prediction
        image = torch.randn(3, 512, 512)
        genomic = torch.randn(config.data.gene_expression_dim)
        
        results = evaluator.predict_patient(image, genomic)
        
        assert 'survival_risk' in results
        assert 'diffusion_tumor' in results
        assert 'tumor_trajectory' in results
        assert results['tumor_trajectory'].shape[0] == config.physics.time_steps
        
        print(f"✓ Evaluation working")
        print(f"  Survival risk: {results['survival_risk']:.4f}")
        print(f"  Trajectory shape: {results['tumor_trajectory'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run complete test suite"""
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Pipeline", test_data_pipeline),
        ("Physics Components", test_physics_components),
        ("Model Architecture", test_model_architecture),
        ("Loss Functions", test_loss_functions),
        ("Training Step", test_training_step),
        ("Evaluation", test_evaluation)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
