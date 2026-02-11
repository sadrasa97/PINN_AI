"""
Main Orchestration Script for Physics-Informed Cancer Modeling
Provides simple interface to run complete pipeline
"""
import os
import argparse
import torch
from config import get_default_config
from data_loader import get_dataloaders
from model import create_model
from train import Trainer
from evaluate import create_analysis_report


def setup_experiment(args):
    """Setup experiment directories and configuration"""
    # Load configuration
    config = get_default_config()
    
    # Override config with command line arguments
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    
    # Create directories
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.tensorboard_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Physics-Informed Neural Network for Cancer Dynamics")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Number of epochs: {config.training.num_epochs}")
    print(f"  Model type: Multi-modal PINN")
    print(f"  Image encoder: {config.model.image_encoder_type}")
    print(f"  Genomic latent dim: {config.model.genomic_latent_dim}")
    print(f"  Physics resolution: {config.physics.spatial_resolution}×{config.physics.spatial_resolution}")
    
    return config


def run_data_preparation(config, args):
    """Prepare or create dataset"""
    print(f"\n{'='*80}")
    print("Data Preparation")
    print(f"{'='*80}\n")
    
    if args.use_synthetic:
        print("Creating synthetic TCGA-like dataset...")
        from data_loader import create_realistic_tcga_data
        create_realistic_tcga_data(
            output_dir="./data"
        )
    else:
        print("Using existing dataset from ./data")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, genomic_scaler = get_dataloaders(
        config, use_real_data=not args.use_synthetic
    )
    
    print(f"\nDataset statistics:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader, genomic_scaler


def run_training(config, args):
    """Run model training"""
    print(f"\n{'='*80}")
    print("Model Training")
    print(f"{'='*80}\n")
    
    # Create trainer
    trainer = Trainer(config, resume_from=args.resume)
    
    # Train
    trainer.train()
    
    return trainer


def run_evaluation(config, trainer, args):
    """Run comprehensive evaluation"""
    print(f"\n{'='*80}")
    print("Model Evaluation")
    print(f"{'='*80}\n")
    
    # Test on best checkpoint
    best_checkpoint = os.path.join(config.training.checkpoint_dir, 'checkpoint_best.pt')
    
    if os.path.exists(best_checkpoint):
        print("Testing with best checkpoint...")
        results, predictions, targets = trainer.test(best_checkpoint)
        
        # Generate analysis report
        if args.generate_report:
            print("\nGenerating comprehensive analysis report...")
            
            # Load best model
            checkpoint = torch.load(best_checkpoint, map_location=config.device)
            model = create_model(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            _, _, test_loader, _ = get_dataloaders(config, create_synthetic=args.use_synthetic)
            
            create_analysis_report(
                model,
                config,
                test_loader,
                save_dir=args.output_dir
            )
    else:
        print("No checkpoint found. Skipping evaluation.")


def run_inference(config, args):
    """Run inference on new data"""
    print(f"\n{'='*80}")
    print("Inference Mode")
    print(f"{'='*80}\n")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    
    print(f"Loaded model from {args.checkpoint}")
    print("Model ready for inference")
    
    # Example inference (would be replaced with actual data loading)
    print("\nExample: Running inference on synthetic patient...")
    
    from evaluate import ModelEvaluator
    evaluator = ModelEvaluator(model, config, config.device)
    
    # Create dummy data for demo
    dummy_image = torch.randn(3, 512, 512)
    dummy_genomic = torch.randn(config.data.gene_expression_dim)
    
    results = evaluator.predict_patient(dummy_image, dummy_genomic)
    
    print("\nPrediction Results:")
    print(f"  Survival Risk: {results['survival_risk']:.4f}")
    print(f"  Progression Probability: {results['progression_prob']:.4f}")
    print(f"  Metastasis Probability: {results['metastasis_prob']:.4f}")
    print(f"\nPhysical Parameters:")
    print(f"  Tumor Diffusion: {results['diffusion_tumor']:.4f} μm²/day")
    print(f"  Tumor Proliferation: {results['proliferation_tumor']:.4f} day⁻¹")
    print(f"  Immune Interaction: {results['interaction_strength']:.4f}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Physics-Informed Neural Network for Cancer Dynamics'
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'train', 'eval', 'infer'],
                       help='Execution mode: full pipeline, train only, eval only, or inference')
    
    # Data arguments
    parser.add_argument('--use_synthetic', action='store_true', default=False,
                       help='Use synthetic TCGA data (default: False, uses real-world data)')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of synthetic samples to generate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default: from config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (default: from config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (default: from config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Evaluation arguments
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint_best.pt',
                       help='Path to checkpoint for evaluation/inference')
    parser.add_argument('--generate_report', action='store_true', default=True,
                       help='Generate comprehensive analysis report')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='Directory for analysis outputs')
    
    args = parser.parse_args()
    
    # Setup
    config = setup_experiment(args)
    
    # Execute based on mode
    if args.mode == 'full':
        # Full pipeline: data prep -> train -> eval
        run_data_preparation(config, args)
        trainer = run_training(config, args)
        run_evaluation(config, trainer, args)
        
    elif args.mode == 'train':
        # Training only
        run_data_preparation(config, args)
        run_training(config, args)
        
    elif args.mode == 'eval':
        # Evaluation only (requires existing checkpoint)
        run_data_preparation(config, args)
        trainer = Trainer(config)
        run_evaluation(config, trainer, args)
        
    elif args.mode == 'infer':
        # Inference mode
        run_inference(config, args)
    
    print(f"\n{'='*80}")
    print("Execution completed successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
