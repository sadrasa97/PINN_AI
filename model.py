"""
Multi-Branch Physics-Informed Neural Network for Cancer Dynamics
Integrates: Image Encoder + Genomic VAE + Physics Module + Multimodal Fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, Optional
import math


class ImageEncoder(nn.Module):
    """
    CNN-based encoder for histopathology images
    Uses pretrained ResNet as backbone
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load pretrained ResNet
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection to output dimension
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, 3, H, W]
        Returns:
            features: [batch, output_dim]
        """
        features = self.backbone(images)
        features = self.projection(features)
        return features


class GenomicVAE(nn.Module):
    """
    Variational Autoencoder for gene expression data
    Learns interpretable latent representation of molecular phenotype
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [1024, 512, 256],
        latent_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed gene expression"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE
        
        Returns:
            reconstruction: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (use mean, no sampling)"""
        mu, _ = self.encode(x)
        return mu


class MultiHeadAttentionFusion(nn.Module):
    """
    Multi-head attention for fusing heterogeneous modalities
    Allows the model to learn which modalities are most relevant for each patient
    """
    
    def __init__(
        self,
        feature_dims: list,
        num_heads: int = 8,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dims: List of feature dimensions for each modality
            num_heads: Number of attention heads
            output_dim: Output dimension after fusion
        """
        super().__init__()
        
        self.num_modalities = len(feature_dims)
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        # Project each modality to a common dimension
        self.modality_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.Dropout(dropout)
        )
        
        self.ffn_layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, modality_features: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_features: List of [batch, feature_dim] tensors
        
        Returns:
            fused_features: [batch, output_dim]
            attention_weights: [batch, num_modalities, num_modalities]
        """
        batch_size = modality_features[0].shape[0]
        
        # Project all modalities to common dimension
        projected = [
            proj(feat) for proj, feat in zip(self.modality_projections, modality_features)
        ]
        
        # Stack into sequence: [batch, num_modalities, output_dim]
        modality_sequence = torch.stack(projected, dim=1)
        
        # Self-attention across modalities
        attn_out, attn_weights = self.attention(
            modality_sequence, modality_sequence, modality_sequence
        )
        
        # Residual connection and layer norm
        attn_out = self.layer_norm(modality_sequence + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_layer_norm(attn_out + ffn_out)
        
        # Pool across modalities (mean pooling)
        fused_features = torch.mean(ffn_out, dim=1)
        
        return fused_features, attn_weights


class ClinicalPredictionHead(nn.Module):
    """
    Multi-task prediction head for clinical outcomes:
    1. Survival analysis (Cox proportional hazards)
    2. Tumor progression (binary classification)
    3. Metastasis prediction (binary classification)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Task-specific heads
        self.survival_head = nn.Linear(prev_dim, 1)  # Cox hazard
        self.progression_head = nn.Linear(prev_dim, 1)  # Binary logit
        self.metastasis_head = nn.Linear(prev_dim, 1)  # Binary logit
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch, input_dim]
        
        Returns:
            predictions: Dictionary with 'survival_risk', 'progression_logit', 'metastasis_logit'
        """
        shared_features = self.shared(features)
        
        return {
            'survival_risk': self.survival_head(shared_features).squeeze(1),
            'progression_logit': self.progression_head(shared_features).squeeze(1),
            'metastasis_logit': self.metastasis_head(shared_features).squeeze(1)
        }


class PhysicsInformedCancerModel(nn.Module):
    """
    Complete Physics-Informed Neural Network for Cancer Dynamics
    
    Architecture:
    1. Image branch: ResNet encoder for histopathology
    2. Genomic branch: VAE for gene expression
    3. Physics branch: Parameter extraction and PDE integration
    4. Fusion: Multi-head attention
    5. Prediction: Multi-task clinical outcomes
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Branch 1: Image encoder
        self.image_encoder = ImageEncoder(
            backbone=config.model.image_encoder_type,
            pretrained=True,
            output_dim=config.model.image_output_dim,
            dropout=config.model.dropout_rate
        )
        
        # Branch 2: Genomic VAE
        self.genomic_vae = GenomicVAE(
            input_dim=config.data.gene_expression_dim,
            hidden_dims=config.model.genomic_encoder_layers,
            latent_dim=config.model.genomic_latent_dim,
            dropout=config.model.dropout_rate
        )
        
        # Multimodal fusion
        self.fusion = MultiHeadAttentionFusion(
            feature_dims=[
                config.model.image_output_dim,
                config.model.genomic_output_dim
            ],
            num_heads=config.model.attention_heads,
            output_dim=config.model.fusion_dim,
            dropout=config.model.dropout_rate
        )
        
        # Branch 3: Physics-informed parameter extraction
        from physics_informed import PhysicalParameterExtractor
        self.physics_extractor = PhysicalParameterExtractor(
            input_dim=config.model.fusion_dim,
            hidden_dims=config.model.physics_hidden_dims,
            num_params=config.model.num_physical_params,
            spatial_resolution=config.physics.spatial_resolution
        )
        
        # Clinical prediction head
        self.clinical_predictor = ClinicalPredictionHead(
            input_dim=config.model.fusion_dim + config.model.num_physical_params,
            hidden_dims=config.model.prediction_hidden_dims,
            dropout=config.model.dropout_rate
        )
    
    def forward(
        self,
        images: torch.Tensor,
        genomic: torch.Tensor,
        return_physics: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model
        
        Args:
            images: [batch, 3, H, W] histopathology images
            genomic: [batch, G] gene expression profiles
            return_physics: Whether to return physics parameters and fields
        
        Returns:
            Dictionary containing:
            - clinical predictions (survival, progression, metastasis)
            - physics parameters (diffusion, proliferation, etc.)
            - initial density fields
            - VAE latent representations
            - attention weights
        """
        batch_size = images.shape[0]
        
        # Encode images
        image_features = self.image_encoder(images)  # [batch, image_dim]
        
        # Encode genomics through VAE
        genomic_recon, genomic_mu, genomic_logvar = self.genomic_vae(genomic)
        genomic_features = genomic_mu  # Use mean as deterministic feature
        
        # Fuse modalities with attention
        fused_features, attention_weights = self.fusion([image_features, genomic_features])
        
        # Extract physical parameters and initial conditions
        physics_params, initial_densities = self.physics_extractor(fused_features)
        
        # Concatenate fused features with physics parameters for clinical prediction
        # Physics parameters provide interpretable biomarkers
        params_concat = torch.stack([
            physics_params['diffusion_tumor'],
            physics_params['diffusion_immune'],
            physics_params['proliferation_tumor'],
            physics_params['proliferation_immune'],
            physics_params['interaction_tumor_immune'],
            physics_params['carrying_capacity']
        ], dim=1)
        
        # Pad to match expected parameter count
        if params_concat.shape[1] < self.config.model.num_physical_params:
            padding = torch.zeros(
                batch_size,
                self.config.model.num_physical_params - params_concat.shape[1],
                device=params_concat.device
            )
            params_concat = torch.cat([params_concat, padding], dim=1)
        
        clinical_features = torch.cat([fused_features, params_concat], dim=1)
        
        # Predict clinical outcomes
        clinical_predictions = self.clinical_predictor(clinical_features)
        
        # Prepare output
        output = {
            'survival_risk': clinical_predictions['survival_risk'],
            'progression_logit': clinical_predictions['progression_logit'],
            'metastasis_logit': clinical_predictions['metastasis_logit'],
            'genomic_reconstruction': genomic_recon,
            'genomic_mu': genomic_mu,
            'genomic_logvar': genomic_logvar,
            'attention_weights': attention_weights,
            'fused_features': fused_features
        }
        
        if return_physics:
            output.update({
                'physics_params': physics_params,
                'initial_densities': initial_densities
            })
        
        return output
    
    def predict_trajectory(
        self,
        images: torch.Tensor,
        genomic: torch.Tensor,
        num_steps: int = 100,
        noise_scale: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate tumor-immune dynamics trajectory using physics model
        
        Returns:
            tumor_trajectory: [batch, num_steps, H, W]
            immune_trajectory: [batch, num_steps, H, W]
        """
        from physics_informed import ReactionDiffusionPDE
        
        # Get physics parameters and initial conditions
        output = self.forward(images, genomic, return_physics=True)
        physics_params = output['physics_params']
        initial_densities = output['initial_densities']
        
        # Split initial densities
        initial_tumor = initial_densities[:, 0:1]
        initial_immune = initial_densities[:, 1:2]
        
        # Create PDE solver
        pde_solver = ReactionDiffusionPDE(
            spatial_resolution=self.config.physics.spatial_resolution,
            dt=self.config.physics.dt
        )
        
        # Simulate trajectory
        tumor_traj, immune_traj = pde_solver.simulate_trajectory(
            initial_tumor, initial_immune, physics_params, num_steps, noise_scale
        )
        
        return tumor_traj, immune_traj


def create_model(config):
    """Factory function to create the PINN model"""
    model = PhysicsInformedCancerModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} parameters ({trainable_params:,} trainable)")
    
    return model


if __name__ == "__main__":
    # Test model architecture
    from config import get_default_config
    
    config = get_default_config()
    model = create_model(config)
    
    # Create dummy batch
    batch_size = 4
    images = torch.randn(batch_size, 3, 512, 512)
    genomic = torch.randn(batch_size, config.data.gene_expression_dim)
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(images, genomic, return_physics=True)
    
    print(f"\nOutput keys: {list(output.keys())}")
    print(f"Survival risk shape: {output['survival_risk'].shape}")
    print(f"Initial densities shape: {output['initial_densities'].shape}")
    print(f"Physics params: {list(output['physics_params'].keys())}")
    
    print("\nTesting trajectory prediction...")
    with torch.no_grad():
        tumor_traj, immune_traj = model.predict_trajectory(images, genomic, num_steps=50)
    
    print(f"Tumor trajectory shape: {tumor_traj.shape}")
    print(f"Immune trajectory shape: {immune_traj.shape}")
    
    print("\nModel architecture test passed!")
