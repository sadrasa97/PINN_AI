"""
Training Pipeline for Physics-Informed Cancer Modeling
Includes: optimization, validation, checkpointing, logging, early stopping
"""
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import json
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from config import get_default_config
from data_loader import get_dataloaders
from model import create_model
from losses import CompletePINNLoss, compute_metrics


class Trainer:
    """
    Complete training pipeline for Physics-Informed Neural Network
    """
    
    def __init__(self, config, resume_from: Optional[str] = None):
        """
        Args:
            config: Configuration object
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        self.set_seed(config.training.seed)
        
        # Create data loaders
        print("Loading data...")
        self.train_loader, self.val_loader, self.test_loader, self.genomic_scaler = get_dataloaders(
            config, use_real_data=True
        )
        
        # Create model
        print("Creating model...")
        self.model = create_model(config).to(self.device)
        
        # Create loss function
        self.criterion = CompletePINNLoss(config).to(self.device)
        
        # Create optimizer
        self.optimizer = self.create_optimizer()
        
        # Create learning rate scheduler
        self.scheduler = self.create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Logging
        os.makedirs(config.training.tensorboard_dir, exist_ok=True)
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(config.training.tensorboard_dir)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Resume from checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        if self.config.training.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def create_optimizer(self):
        """Create optimizer based on config"""
        if self.config.training.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        return optimizer
    
    def create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.training.scheduler.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=1e-6
            )
        elif self.config.training.scheduler.lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of average training metrics
        """
        self.model.train()
        
        epoch_losses = {
            'total': [], 'clinical_total': [], 'vae_total': [], 'physics_total': [],
            'survival': [], 'progression': [], 'metastasis': [],
            'vae_reconstruction': [], 'vae_kl': [],
            'pde_residual': [], 'boundary': [], 'entropy': [], 'sparsity': []
        }
        
        epoch_metrics = {'c_index': [], 'progression_auroc': [], 'metastasis_auroc': []}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.training.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            genomic = batch['genomic'].to(self.device)
            
            targets = {
                'survival_time': batch['survival_time'].to(self.device),
                'survival_event': batch['survival_event'].to(self.device),
                'progression': batch['progression'].to(self.device),
                'metastasis': batch['metastasis'].to(self.device),
                'genomic': genomic
            }
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    output = self.model(images, genomic, return_physics=True)
                    loss, loss_dict = self.criterion(output, targets, compute_physics=True)
            else:
                output = self.model(images, genomic, return_physics=True)
                loss, loss_dict = self.criterion(output, targets, compute_physics=True)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )
                self.optimizer.step()
            
            # Record losses
            for key in epoch_losses.keys():
                if key in loss_dict:
                    epoch_losses[key].append(loss_dict[key])
            
            # Compute metrics
            with torch.no_grad():
                batch_metrics = compute_metrics(output, targets)
                for key in epoch_metrics.keys():
                    if key in batch_metrics:
                        epoch_metrics[key].append(batch_metrics[key])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'c_idx': f"{batch_metrics.get('c_index', 0):.3f}"
            })
            
            # Log to tensorboard
            if self.global_step % self.config.training.log_interval == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
                
                self.writer.add_scalar('train/learning_rate', 
                                     self.optimizer.param_groups[0]['lr'], 
                                     self.global_step)
            
            self.global_step += 1
        
        # Average metrics over epoch
        avg_losses = {key: np.mean(values) if values else 0.0 
                     for key, values in epoch_losses.items()}
        avg_metrics = {key: np.mean(values) if values else 0.0 
                      for key, values in epoch_metrics.items()}
        
        return {**avg_losses, **avg_metrics}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate on validation set
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        val_losses = {
            'total': [], 'clinical_total': [], 'vae_total': [], 'physics_total': [],
            'survival': [], 'progression': [], 'metastasis': [],
            'vae_reconstruction': [], 'vae_kl': [],
            'pde_residual': [], 'boundary': [], 'entropy': [], 'sparsity': []
        }
        
        val_metrics = {'c_index': [], 'progression_auroc': [], 'metastasis_auroc': []}
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            genomic = batch['genomic'].to(self.device)
            
            targets = {
                'survival_time': batch['survival_time'].to(self.device),
                'survival_event': batch['survival_event'].to(self.device),
                'progression': batch['progression'].to(self.device),
                'metastasis': batch['metastasis'].to(self.device),
                'genomic': genomic
            }
            
            # Forward pass
            output = self.model(images, genomic, return_physics=True)
            loss, loss_dict = self.criterion(output, targets, compute_physics=True)
            
            # Record losses
            for key in val_losses.keys():
                if key in loss_dict:
                    val_losses[key].append(loss_dict[key])
            
            # Compute metrics
            batch_metrics = compute_metrics(output, targets)
            for key in val_metrics.keys():
                if key in batch_metrics:
                    val_metrics[key].append(batch_metrics[key])
        
        # Average metrics
        avg_losses = {key: np.mean(values) if values else 0.0 
                     for key, values in val_losses.items()}
        avg_metrics = {key: np.mean(values) if values else 0.0 
                      for key, values in val_metrics.items()}
        
        # Log to tensorboard
        for key, value in {**avg_losses, **avg_metrics}.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        return {**avg_losses, **avg_metrics}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.training.checkpoint_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.training.checkpoint_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {metrics['total']:.4f})")
        
        # Save periodic checkpoint
        if epoch % 20 == 0:
            epoch_path = os.path.join(self.config.training.checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self):
        """
        Main training loop
        """
        print(f"\n{'='*80}")
        print("Starting training...")
        print(f"{'='*80}\n")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch} - Training:")
            print(f"  Loss: {train_metrics['total']:.4f}")
            print(f"  C-index: {train_metrics['c_index']:.4f}")
            print(f"  Progression AUROC: {train_metrics['progression_auroc']:.4f}")
            print(f"  Metastasis AUROC: {train_metrics['metastasis_auroc']:.4f}")
            
            # Validate
            if epoch % self.config.training.val_interval == 0:
                val_metrics = self.validate(epoch)
                
                print(f"\nEpoch {epoch} - Validation:")
                print(f"  Loss: {val_metrics['total']:.4f}")
                print(f"  C-index: {val_metrics['c_index']:.4f}")
                print(f"  Progression AUROC: {val_metrics['progression_auroc']:.4f}")
                print(f"  Metastasis AUROC: {val_metrics['metastasis_auroc']:.4f}")
                
                # Check for improvement
                is_best = val_metrics['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['total']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Early stopping
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['total'])
                    else:
                        self.scheduler.step()
        
        print(f"\n{'='*80}")
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*80}\n")
        
        self.writer.close()
    
    @torch.no_grad()
    def test(self, checkpoint_path: Optional[str] = None):
        """
        Evaluate on test set
        
        Args:
            checkpoint_path: Path to checkpoint to load (if None, use current model)
        """
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        test_losses = {
            'total': [], 'clinical_total': [], 'vae_total': [], 'physics_total': [],
            'survival': [], 'progression': [], 'metastasis': []
        }
        
        test_metrics = {'c_index': [], 'progression_auroc': [], 'metastasis_auroc': [],
                       'progression_accuracy': [], 'metastasis_accuracy': []}
        
        all_predictions = []
        all_targets = []
        
        for batch in tqdm(self.test_loader, desc='Testing'):
            images = batch['image'].to(self.device)
            genomic = batch['genomic'].to(self.device)
            
            targets = {
                'survival_time': batch['survival_time'].to(self.device),
                'survival_event': batch['survival_event'].to(self.device),
                'progression': batch['progression'].to(self.device),
                'metastasis': batch['metastasis'].to(self.device),
                'genomic': genomic
            }
            
            # Forward pass
            output = self.model(images, genomic, return_physics=True)
            loss, loss_dict = self.criterion(output, targets, compute_physics=True)
            
            # Record losses
            for key in test_losses.keys():
                if key in loss_dict:
                    test_losses[key].append(loss_dict[key])
            
            # Compute metrics
            batch_metrics = compute_metrics(output, targets)
            for key in test_metrics.keys():
                if key in batch_metrics:
                    test_metrics[key].append(batch_metrics[key])
            
            # Store predictions
            all_predictions.append(output)
            all_targets.append(targets)
        
        # Average metrics
        avg_losses = {key: np.mean(values) if values else 0.0 
                     for key, values in test_losses.items()}
        avg_metrics = {key: np.mean(values) if values else 0.0 
                      for key, values in test_metrics.items()}
        
        # Print results
        print(f"\n{'='*80}")
        print("Test Set Results:")
        print(f"{'='*80}")
        print(f"Loss: {avg_losses['total']:.4f}")
        print(f"C-index (Survival): {avg_metrics['c_index']:.4f}")
        print(f"Progression AUROC: {avg_metrics['progression_auroc']:.4f}")
        print(f"Progression Accuracy: {avg_metrics['progression_accuracy']:.4f}")
        print(f"Metastasis AUROC: {avg_metrics['metastasis_auroc']:.4f}")
        print(f"Metastasis Accuracy: {avg_metrics['metastasis_accuracy']:.4f}")
        print(f"{'='*80}\n")
        
        # Save results
        results = {**avg_losses, **avg_metrics}
        results_path = os.path.join(self.config.training.checkpoint_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_path}")
        
        return results, all_predictions, all_targets


def main():
    """Main training function"""
    # Load configuration
    config = get_default_config()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train model
    trainer.train()
    
    # Test on best checkpoint
    best_checkpoint = os.path.join(config.training.checkpoint_dir, 'checkpoint_best.pt')
    if os.path.exists(best_checkpoint):
        print("\nEvaluating best model on test set...")
        trainer.test(best_checkpoint)


if __name__ == "__main__":
    main()
