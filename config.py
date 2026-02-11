"""
Configuration Management for Physics-Informed Neural Network Cancer Modeling Framework
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class DataConfig:
    """TCGA-like multimodal dataset configuration"""
    # Data paths
    image_dir: str = "./data/histopathology"
    genomic_file: str = "./data/transcriptomics.csv"
    clinical_file: str = "./data/clinical_outcomes.csv"
    
    # Image preprocessing
    image_size: Tuple[int, int] = (512, 512)
    patch_size: int = 256
    color_normalization: str = "macenko"  # Macenko method for H&E staining
    
    # Genomic data
    gene_expression_dim: int = 20530  # Typical RNA-seq dimension (TCGA level 3 data)
    genomic_latent_dim: int = 256
    
    # Clinical targets
    survival_target: str = "overall_survival"
    progression_target: str = "tumor_progression"
    metastasis_target: str = "metastatic_status"
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    # Image encoder (ResNet-based CNN)
    image_encoder_type: str = "resnet34"
    image_feature_dim: int = 512
    image_output_dim: int = 512
    
    # Genomic encoder (VAE)
    genomic_encoder_layers: List[int] = field(default_factory=lambda: [1024, 512, 256])
    genomic_latent_dim: int = 256
    genomic_output_dim: int = 256
    vae_beta: float = 1.0  # KL divergence weight
    
    # Physics-informed component
    physics_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    physics_output_dim: int = 128
    num_physical_params: int = 8  # Diffusion, proliferation, interaction params
    
    # Multimodal fusion
    fusion_type: str = "attention"  # "attention" or "concat"
    fusion_dim: int = 512
    attention_heads: int = 8
    
    # Final prediction layers
    prediction_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    num_clinical_outputs: int = 3  # Survival, progression, metastasis
    
    # Dropout and regularization
    dropout_rate: float = 0.3
    use_batch_norm: bool = True


@dataclass
class PhysicsConfig:
    """Statistical physics and biophysics modeling parameters"""
    # Reaction-diffusion equation parameters
    spatial_dim: int = 2  # 2D tissue modeling
    num_cell_types: int = 2  # Tumor cells and immune cells
    
    # Physical parameter bounds (for interpretation and constraints)
    diffusion_coeff_min: float = 0.01  # μm²/day
    diffusion_coeff_max: float = 100.0
    proliferation_rate_min: float = 0.0  # day⁻¹
    proliferation_rate_max: float = 2.0
    interaction_strength_min: float = -10.0  # Attractive/repulsive
    interaction_strength_max: float = 10.0
    
    # Stochastic dynamics
    temperature: float = 1.0  # Effective temperature for Langevin dynamics
    noise_scale: float = 0.1
    
    # Active matter parameters
    activity_coefficient: float = 1.0
    alignment_strength: float = 0.5
    
    # PDE discretization
    spatial_resolution: int = 64  # Grid points per dimension
    time_steps: int = 100
    dt: float = 0.1  # Time step for numerical integration
    
    # Physics loss weights
    pde_residual_weight: float = 1.0
    boundary_condition_weight: float = 0.5
    entropy_regularization_weight: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters and optimization settings"""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # "cosine" or "plateau"
    
    # Training duration
    num_epochs: int = 200
    early_stopping_patience: int = 30
    
    # Loss function weights
    clinical_loss_weight: float = 1.0
    physics_loss_weight: float = 0.5
    vae_kl_weight: float = 0.1
    entropy_reg_weight: float = 0.05
    sparsity_weight: float = 0.01
    
    # Gradient management
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    
    # Logging and checkpointing
    log_interval: int = 10
    val_interval: int = 1
    checkpoint_dir: str = "./checkpoints"
    tensorboard_dir: str = "./runs"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation metrics and validation settings"""
    # Clinical prediction metrics
    survival_metrics: List[str] = field(default_factory=lambda: [
        "concordance_index", "brier_score", "calibration_slope"
    ])
    classification_metrics: List[str] = field(default_factory=lambda: [
        "auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1_score"
    ])
    
    # Physics validation
    check_pde_residuals: bool = True
    check_parameter_stability: bool = True
    check_energy_conservation: bool = True
    
    # Visualization
    plot_survival_curves: bool = True
    plot_parameter_distributions: bool = True
    plot_attention_maps: bool = True
    
    # Statistical testing
    num_bootstrap_samples: int = 1000
    confidence_level: float = 0.95


@dataclass
class Config:
    """Master configuration object"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus: int = torch.cuda.device_count()
    
    def __post_init__(self):
        """Validate configuration consistency"""
        assert self.data.train_ratio + self.data.val_ratio + self.data.test_ratio == 1.0
        assert self.model.genomic_latent_dim == self.model.genomic_output_dim
        assert self.physics.spatial_dim in [1, 2, 3]
        assert self.physics.num_cell_types >= 1


def get_default_config() -> Config:
    """Return default configuration for experiments"""
    return Config()


if __name__ == "__main__":
    # Test configuration
    config = get_default_config()
    print("Configuration loaded successfully")
    print(f"Device: {config.device}")
    print(f"Number of GPUs: {config.num_gpus}")
