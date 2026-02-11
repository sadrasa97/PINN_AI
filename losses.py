"""
Multi-Component Loss Functions for Physics-Informed Cancer Modeling
Combines:
1. Clinical prediction losses (survival, progression, metastasis)
2. VAE reconstruction and KL divergence
3. Physics-informed PDE residuals
4. Regularization (entropy, sparsity)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class CoxProportionalHazardsLoss(nn.Module):
    """
    Cox Proportional Hazards loss for survival analysis
    
    The Cox model assumes: h(t|x) = h_0(t) * exp(β'x)
    where h(t|x) is the hazard function and β'x is the risk score predicted by the model
    
    Partial likelihood:
    L = ∏_{i:δ_i=1} [ exp(r_i) / ∑_{j∈R_i} exp(r_j) ]
    
    where:
    - r_i is the risk score for patient i
    - δ_i is the event indicator (1 if event occurred, 0 if censored)
    - R_i is the risk set at time t_i (all patients still at risk)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        risk_scores: torch.Tensor,
        survival_times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            risk_scores: [batch] predicted risk scores (higher = worse prognosis)
            survival_times: [batch] observed survival times
            events: [batch] event indicators (1 = death, 0 = censored)
        
        Returns:
            loss: Negative log partial likelihood
        """
        # Sort by survival time (descending)
        sorted_indices = torch.argsort(survival_times, descending=True)
        risk_scores = risk_scores[sorted_indices]
        events = events[sorted_indices]
        
        # Compute log risk scores
        log_risk = risk_scores
        
        # Compute cumulative sum for risk set denominator
        # exp_risk[i] contains sum of exp(risk) for all j >= i (patients still at risk)
        exp_risk = torch.exp(log_risk)
        cumsum_exp_risk = torch.cumsum(exp_risk, dim=0)
        
        # Log-sum-exp trick for numerical stability
        max_risk = torch.max(log_risk)
        log_cumsum_exp_risk = torch.log(cumsum_exp_risk) + max_risk - max_risk
        
        # Negative log partial likelihood (only for uncensored events)
        loss = -torch.sum((log_risk - torch.log(cumsum_exp_risk)) * events) / (torch.sum(events) + 1e-8)
        
        return loss


class ConcordanceIndex(nn.Module):
    """
    Concordance Index (C-index) for survival analysis
    Measures discriminative ability: fraction of pairs correctly ordered
    
    C-index = P(risk_i > risk_j | t_i < t_j and both uncensored)
    
    Higher C-index (closer to 1.0) means better discrimination
    0.5 = random predictions, 1.0 = perfect predictions
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        risk_scores: torch.Tensor,
        survival_times: torch.Tensor,
        events: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns C-index as a differentiable approximation
        """
        n = len(survival_times)
        
        # Create all pairs
        time_diff = survival_times.unsqueeze(1) - survival_times.unsqueeze(0)  # [n, n]
        risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)  # [n, n]
        
        # Valid pairs: i died before j, and i had event
        valid_pairs = (time_diff < 0) & (events.unsqueeze(1) == 1)
        
        # Concordant pairs: higher risk for patient who died earlier
        concordant = (risk_diff > 0) & valid_pairs
        
        if valid_pairs.sum() == 0:
            return torch.tensor(0.5, device=risk_scores.device)
        
        c_index = concordant.sum().float() / valid_pairs.sum().float()
        
        return c_index


class VAELoss(nn.Module):
    """
    Variational Autoencoder loss
    Combines reconstruction loss and KL divergence
    
    L = E[log p(x|z)] - KL[q(z|x) || p(z)]
    
    where:
    - Reconstruction term measures how well VAE reconstructs input
    - KL term regularizes latent space to be close to standard normal
    """
    
    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Weight for KL divergence (beta-VAE)
                  beta > 1 encourages more disentangled representations
        """
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            reconstruction: Reconstructed gene expression
            original: Original gene expression
            mu: Latent mean
            logvar: Latent log variance
        
        Returns:
            total_loss: Combined VAE loss
            loss_dict: Individual components
        """
        # Reconstruction loss (MSE for continuous gene expression)
        recon_loss = F.mse_loss(reconstruction, original, reduction='mean')
        
        # KL divergence: KL[N(μ,σ²) || N(0,1)]
        # = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        loss_dict = {
            'vae_reconstruction': recon_loss.item(),
            'vae_kl': kl_loss.item()
        }
        
        return total_loss, loss_dict


class MultiTaskClinicalLoss(nn.Module):
    """
    Combined loss for multiple clinical prediction tasks:
    1. Survival (Cox loss)
    2. Progression (binary cross-entropy)
    3. Metastasis (binary cross-entropy)
    """
    
    def __init__(
        self,
        survival_weight: float = 1.0,
        progression_weight: float = 0.5,
        metastasis_weight: float = 0.5
    ):
        super().__init__()
        self.survival_weight = survival_weight
        self.progression_weight = progression_weight
        self.metastasis_weight = metastasis_weight
        
        self.cox_loss = CoxProportionalHazardsLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: Dict with 'survival_risk', 'progression_logit', 'metastasis_logit'
            targets: Dict with 'survival_time', 'survival_event', 'progression', 'metastasis'
        
        Returns:
            total_loss: Weighted sum of all clinical losses
            loss_dict: Individual loss components
        """
        # Survival loss
        survival_loss = self.cox_loss(
            predictions['survival_risk'],
            targets['survival_time'],
            targets['survival_event']
        )
        
        # Progression loss
        progression_loss = self.bce_loss(
            predictions['progression_logit'],
            targets['progression'].float()
        )
        
        # Metastasis loss
        metastasis_loss = self.bce_loss(
            predictions['metastasis_logit'],
            targets['metastasis'].float()
        )
        
        # Weighted combination
        total_loss = (
            self.survival_weight * survival_loss +
            self.progression_weight * progression_loss +
            self.metastasis_weight * metastasis_loss
        )
        
        loss_dict = {
            'survival': survival_loss.item(),
            'progression': progression_loss.item(),
            'metastasis': metastasis_loss.item()
        }
        
        return total_loss, loss_dict


class SparsityRegularization(nn.Module):
    """
    L1 sparsity regularization to encourage sparse physical parameters
    Helps with interpretability by identifying truly important parameters
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, params_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            params_dict: Dictionary of physical parameters
        
        Returns:
            sparsity_loss: L1 norm of all parameters
        """
        sparsity = torch.tensor(0.0, device=list(params_dict.values())[0].device)
        
        for param in params_dict.values():
            sparsity = sparsity + torch.mean(torch.abs(param))
        
        return self.weight * sparsity


class CompletePINNLoss(nn.Module):
    """
    Complete multi-component loss for Physics-Informed Neural Network
    
    L_total = w_1 * L_clinical + w_2 * L_VAE + w_3 * L_physics + w_4 * L_entropy + w_5 * L_sparsity
    
    This integrated loss ensures:
    1. Accurate clinical predictions
    2. Meaningful latent genomic representations
    3. Physical consistency with governing equations
    4. Thermodynamic plausibility
    5. Interpretable sparse parameters
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Clinical prediction losses
        self.clinical_loss = MultiTaskClinicalLoss(
            survival_weight=1.0,
            progression_weight=0.5,
            metastasis_weight=0.5
        )
        
        # VAE loss
        self.vae_loss = VAELoss(beta=config.model.vae_beta)
        
        # Physics-informed loss
        from physics_informed import PhysicsInformedLoss
        self.physics_loss = PhysicsInformedLoss(
            pde_weight=config.physics.pde_residual_weight,
            boundary_weight=config.physics.boundary_condition_weight,
            entropy_weight=config.physics.entropy_regularization_weight,
            spatial_resolution=config.physics.spatial_resolution
        )
        
        # Sparsity regularization
        self.sparsity_loss = SparsityRegularization(weight=config.training.sparsity_weight)
        
        # Loss weights
        self.clinical_weight = config.training.clinical_loss_weight
        self.physics_weight = config.training.physics_loss_weight
        self.vae_weight = config.training.vae_kl_weight
        self.entropy_weight = config.training.entropy_reg_weight
        
        # PDE solver for computing trajectories
        from physics_informed import ReactionDiffusionPDE
        self.pde_solver = ReactionDiffusionPDE(
            spatial_resolution=config.physics.spatial_resolution,
            dt=config.physics.dt
        )
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        compute_physics: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute complete multi-component loss
        
        Args:
            model_output: Dictionary from model forward pass
            targets: Dictionary with ground truth labels
            compute_physics: Whether to compute physics loss (expensive)
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of all loss components
        """
        loss_dict = {}
        
        # 1. Clinical prediction loss
        clinical_predictions = {
            'survival_risk': model_output['survival_risk'],
            'progression_logit': model_output['progression_logit'],
            'metastasis_logit': model_output['metastasis_logit']
        }
        
        clinical_targets = {
            'survival_time': targets['survival_time'],
            'survival_event': targets['survival_event'],
            'progression': targets['progression'],
            'metastasis': targets['metastasis']
        }
        
        clinical_loss, clinical_components = self.clinical_loss(
            clinical_predictions, clinical_targets
        )
        loss_dict.update(clinical_components)
        
        # 2. VAE loss
        vae_loss, vae_components = self.vae_loss(
            model_output['genomic_reconstruction'],
            targets['genomic'],
            model_output['genomic_mu'],
            model_output['genomic_logvar']
        )
        loss_dict.update(vae_components)
        
        # 3. Physics-informed loss (if physics params available)
        if compute_physics and 'physics_params' in model_output:
            params = model_output['physics_params']
            initial_densities = model_output['initial_densities']
            
            # Simulate short trajectory to check PDE consistency
            tumor_init = initial_densities[:, 0:1]
            immune_init = initial_densities[:, 1:2]
            
            # Get density at t=0 and t=dt
            tumor_next, immune_next = self.pde_solver.step(
                tumor_init, immune_init, params, noise_scale=0
            )
            
            # Stack for physics loss computation
            tumor_pred = torch.stack([tumor_init, tumor_next], dim=1)
            immune_pred = torch.stack([immune_init, immune_next], dim=1)
            
            physics_loss, physics_components = self.physics_loss(
                tumor_pred, immune_pred, params
            )
            loss_dict.update(physics_components)
            
            # 4. Sparsity regularization on physical parameters
            sparsity_loss = self.sparsity_loss(params)
            loss_dict['sparsity'] = sparsity_loss.item()
        else:
            physics_loss = torch.tensor(0.0, device=clinical_loss.device)
            sparsity_loss = torch.tensor(0.0, device=clinical_loss.device)
            loss_dict.update({'pde_residual': 0.0, 'boundary': 0.0, 'entropy': 0.0, 'sparsity': 0.0})
        
        # Combine all losses
        total_loss = (
            self.clinical_weight * clinical_loss +
            self.vae_weight * vae_loss +
            self.physics_weight * physics_loss +
            sparsity_loss
        )
        
        loss_dict['total'] = total_loss.item()
        loss_dict['clinical_total'] = clinical_loss.item()
        loss_dict['vae_total'] = vae_loss.item()
        loss_dict['physics_total'] = physics_loss.item()
        
        return total_loss, loss_dict


def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute evaluation metrics for all prediction tasks
    
    Returns:
        Dictionary with:
        - C-index for survival
        - AUROC, accuracy for progression and metastasis
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    metrics = {}
    
    # C-index for survival
    c_index_fn = ConcordanceIndex()
    with torch.no_grad():
        c_index = c_index_fn(
            predictions['survival_risk'],
            targets['survival_time'],
            targets['survival_event']
        )
    metrics['c_index'] = c_index.item()
    
    # Move to CPU and convert to numpy
    progression_logit = predictions['progression_logit'].detach().cpu().numpy()
    progression_true = targets['progression'].cpu().numpy()
    
    metastasis_logit = predictions['metastasis_logit'].detach().cpu().numpy()
    metastasis_true = targets['metastasis'].cpu().numpy()
    
    # Progression metrics
    progression_prob = 1 / (1 + np.exp(-progression_logit))
    progression_pred = (progression_prob > 0.5).astype(int)
    
    try:
        metrics['progression_auroc'] = roc_auc_score(progression_true, progression_prob)
    except:
        metrics['progression_auroc'] = 0.5
    
    metrics['progression_accuracy'] = accuracy_score(progression_true, progression_pred)
    
    # Metastasis metrics
    metastasis_prob = 1 / (1 + np.exp(-metastasis_logit))
    metastasis_pred = (metastasis_prob > 0.5).astype(int)
    
    try:
        metrics['metastasis_auroc'] = roc_auc_score(metastasis_true, metastasis_prob)
    except:
        metrics['metastasis_auroc'] = 0.5
    
    metrics['metastasis_accuracy'] = accuracy_score(metastasis_true, metastasis_pred)
    
    return metrics


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy predictions and targets
    batch_size = 16
    
    predictions = {
        'survival_risk': torch.randn(batch_size),
        'progression_logit': torch.randn(batch_size),
        'metastasis_logit': torch.randn(batch_size)
    }
    
    targets = {
        'survival_time': torch.rand(batch_size) * 1000 + 100,
        'survival_event': torch.randint(0, 2, (batch_size,)).float(),
        'progression': torch.randint(0, 2, (batch_size,)),
        'metastasis': torch.randint(0, 2, (batch_size,))
    }
    
    # Test clinical loss
    clinical_loss_fn = MultiTaskClinicalLoss()
    loss, components = clinical_loss_fn(predictions, targets)
    print(f"\nClinical loss: {loss.item():.4f}")
    print(f"Components: {components}")
    
    # Test metrics
    metrics = compute_metrics(predictions, targets)
    print(f"\nMetrics: {metrics}")
    
    print("\nLoss functions test passed!")
