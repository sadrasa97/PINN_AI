"""
Physics-Informed Components for Cancer Dynamics Modeling
Implements:
1. Reaction-diffusion equations for tumor growth
2. Stochastic Langevin dynamics
3. Active matter formulations
4. PDE residual computation for physics-informed loss
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional


class ReactionDiffusionPDE:
    """
    Reaction-diffusion system for tumor-immune dynamics
    
    System of PDEs:
    ∂u/∂t = D_u ∇²u + r_u u(1 - u/K) - α u v + η_u
    ∂v/∂t = D_v ∇²v + r_v v - β u v + η_v
    
    where:
    u = tumor cell density
    v = immune cell density
    D_u, D_v = diffusion coefficients
    r_u, r_v = proliferation rates
    K = carrying capacity
    α = tumor killing by immune cells
    β = immune activation by tumor
    η = stochastic noise terms
    """
    
    def __init__(
        self,
        spatial_resolution: int = 64,
        num_cell_types: int = 2,
        dt: float = 0.1,
        dx: float = 1.0
    ):
        """
        Args:
            spatial_resolution: Grid points per spatial dimension
            num_cell_types: Number of interacting populations (tumor, immune, etc.)
            dt: Time step
            dx: Spatial step
        """
        self.resolution = spatial_resolution
        self.num_types = num_cell_types
        self.dt = dt
        self.dx = dx
    
    def laplacian_2d(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D Laplacian using finite differences

        Args:
            field: [batch, 1, H, W] density field
        Returns:
            laplacian: [batch, 1, H, W]
        """
        # Handle the specific case mentioned in the error: 5D tensor
        if len(field.shape) == 5:
            # The most likely scenario is that we have [batch, 1, 1, H, W] or similar
            # Try to reshape to 4D by squeezing singleton dimensions
            original_shape = field.shape
            reshaped_field = field
            
            # Squeeze all singleton dimensions if possible
            dims_to_squeeze = []
            for i, size in enumerate(original_shape):
                if size == 1:
                    dims_to_squeeze.append(i)
                    
            # Squeeze from the end backwards to avoid index shifting
            for dim in sorted(dims_to_squeeze, reverse=True):
                reshaped_field = torch.squeeze(reshaped_field, dim=dim)
                
            # If we still have 5D, we need to handle it differently
            if len(reshaped_field.shape) == 5:
                # Take first element along first dimension to make it 4D
                reshaped_field = reshaped_field[0]
            elif len(reshaped_field.shape) < 4:
                # If we went too far, unsqueeze to get back to 4D
                while len(reshaped_field.shape) < 4:
                    reshaped_field = reshaped_field.unsqueeze(0)
            
            field = reshaped_field
        elif len(field.shape) != 4:
            # If tensor is not 4D and not 5D, raise an error
            raise ValueError(f"Expected 4D tensor [batch, channels, H, W], got {field.shape}")
        
        # Pad for boundary conditions (Neumann: zero flux)
        field_pad = torch.nn.functional.pad(field, (1, 1, 1, 1), mode='replicate')

        # Finite difference stencil
        laplacian = (
            field_pad[:, :, 2:, 1:-1] +    # top
            field_pad[:, :, :-2, 1:-1] +   # bottom
            field_pad[:, :, 1:-1, 2:] +    # right
            field_pad[:, :, 1:-1, :-2] -   # left
            4 * field[:, :, :, :]          # center
        ) / (self.dx ** 2)

        return laplacian
    
    def compute_reaction_terms(
        self,
        tumor_density: torch.Tensor,
        immune_density: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reaction terms for tumor-immune interactions
        
        Args:
            tumor_density: [batch, 1, H, W]
            immune_density: [batch, 1, H, W]
            params: Dictionary with 'proliferation_tumor', 'proliferation_immune',
                   'carrying_capacity', 'interaction_tumor_immune'
        
        Returns:
            tumor_reaction: [batch, 1, H, W]
            immune_reaction: [batch, 1, H, W]
        """
        r_tumor = params['proliferation_tumor'].view(-1, 1, 1, 1)
        r_immune = params['proliferation_immune'].view(-1, 1, 1, 1)
        K = params['carrying_capacity'].view(-1, 1, 1, 1)
        alpha = params['interaction_tumor_immune'].view(-1, 1, 1, 1)
        
        # Tumor reaction: logistic growth - immune killing
        tumor_reaction = r_tumor * tumor_density * (1 - tumor_density / K) - alpha * tumor_density * immune_density
        
        # Immune reaction: activation by tumor presence - natural decay
        beta = 0.1  # Immune activation coefficient
        immune_reaction = beta * tumor_density * immune_density - 0.05 * immune_density
        
        return tumor_reaction, immune_reaction
    
    def step(
        self,
        tumor_density: torch.Tensor,
        immune_density: torch.Tensor,
        params: Dict[str, torch.Tensor],
        noise_scale: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one time step of the reaction-diffusion system
        
        Args:
            tumor_density: Current tumor density [batch, 1, H, W]
            immune_density: Current immune density [batch, 1, H, W]
            params: Physical parameters
            noise_scale: Stochastic noise amplitude
        
        Returns:
            tumor_next: Updated tumor density
            immune_next: Updated immune density
        """
        D_tumor = params['diffusion_tumor'].view(-1, 1, 1, 1)
        D_immune = params['diffusion_immune'].view(-1, 1, 1, 1)
        
        # Diffusion terms
        tumor_diffusion = D_tumor * self.laplacian_2d(tumor_density)
        immune_diffusion = D_immune * self.laplacian_2d(immune_density)
        
        # Reaction terms
        tumor_reaction, immune_reaction = self.compute_reaction_terms(
            tumor_density, immune_density, params
        )
        
        # Stochastic noise (Langevin dynamics)
        if noise_scale > 0:
            tumor_noise = noise_scale * torch.randn_like(tumor_density)
            immune_noise = noise_scale * torch.randn_like(immune_density)
        else:
            tumor_noise = 0
            immune_noise = 0
        
        # Forward Euler integration
        tumor_next = tumor_density + self.dt * (tumor_diffusion + tumor_reaction + tumor_noise)
        immune_next = immune_density + self.dt * (immune_diffusion + immune_reaction + immune_noise)
        
        # Ensure non-negativity (physical constraint)
        tumor_next = torch.clamp(tumor_next, min=0)
        immune_next = torch.clamp(immune_next, min=0)
        
        return tumor_next, immune_next
    
    def simulate_trajectory(
        self,
        initial_tumor: torch.Tensor,
        initial_immune: torch.Tensor,
        params: Dict[str, torch.Tensor],
        num_steps: int = 100,
        noise_scale: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate full trajectory of the system
        
        Returns:
            tumor_trajectory: [batch, num_steps, H, W]
            immune_trajectory: [batch, num_steps, H, W]
        """
        batch_size = initial_tumor.shape[0]
        H, W = initial_tumor.shape[2], initial_tumor.shape[3]
        
        tumor_traj = torch.zeros(batch_size, num_steps, H, W, device=initial_tumor.device)
        immune_traj = torch.zeros(batch_size, num_steps, H, W, device=initial_immune.device)
        
        tumor_current = initial_tumor
        immune_current = initial_immune
        
        for t in range(num_steps):
            tumor_traj[:, t] = tumor_current.squeeze(1)
            immune_traj[:, t] = immune_current.squeeze(1)
            
            tumor_current, immune_current = self.step(
                tumor_current, immune_current, params, noise_scale
            )
        
        return tumor_traj, immune_traj


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function combining:
    1. PDE residuals (how well does the model satisfy governing equations)
    2. Boundary conditions
    3. Initial conditions
    4. Entropy regularization (thermodynamic consistency)
    """
    
    def __init__(
        self,
        pde_weight: float = 1.0,
        boundary_weight: float = 0.5,
        entropy_weight: float = 0.1,
        spatial_resolution: int = 64
    ):
        super().__init__()
        self.pde_weight = pde_weight
        self.boundary_weight = boundary_weight
        self.entropy_weight = entropy_weight
        self.pde_solver = ReactionDiffusionPDE(spatial_resolution=spatial_resolution)
    
    def compute_pde_residual(
        self,
        tumor_pred: torch.Tensor,
        immune_pred: torch.Tensor,
        params: Dict[str, torch.Tensor],
        dt: float = 0.1
    ) -> torch.Tensor:
        """
        Compute residual of the reaction-diffusion PDE
        
        The residual measures how much the predicted densities violate the PDE.
        Lower residual = better physical consistency
        
        Args:
            tumor_pred: Predicted tumor density at time t and t+dt [batch, 2, H, W]
            immune_pred: Predicted immune density [batch, 2, H, W]
            params: Physical parameters
            dt: Time step
        
        Returns:
            residual: Scalar PDE residual loss
        """
        tumor_t = tumor_pred[:, 0:1]
        tumor_t_next = tumor_pred[:, 1:2]
        immune_t = immune_pred[:, 0:1]
        immune_t_next = immune_pred[:, 1:2]
        
        # Compute what the PDE predicts for the next time step
        tumor_pde, immune_pde = self.pde_solver.step(tumor_t, immune_t, params, noise_scale=0)
        
        # Residual = difference between model prediction and PDE prediction
        tumor_residual = torch.mean((tumor_t_next - tumor_pde) ** 2)
        immune_residual = torch.mean((immune_t_next - immune_pde) ** 2)
        
        return tumor_residual + immune_residual
    
    def compute_boundary_loss(
        self,
        density_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce no-flux boundary conditions
        Neumann BC: ∂u/∂n = 0 at boundaries
        """
        # Check that boundary values don't change rapidly (soft constraint)
        top = density_field[:, :, 0, :]
        bottom = density_field[:, :, -1, :]
        left = density_field[:, :, :, 0]
        right = density_field[:, :, :, -1]
        
        # Penalize large gradients at boundaries
        boundary_loss = (
            torch.mean(torch.abs(top[:, :, 1:] - top[:, :, :-1])) +
            torch.mean(torch.abs(bottom[:, :, 1:] - bottom[:, :, :-1])) +
            torch.mean(torch.abs(left[:, :, 1:] - left[:, :, :-1])) +
            torch.mean(torch.abs(right[:, :, 1:] - right[:, :, :-1]))
        )
        
        return boundary_loss
    
    def compute_entropy_regularization(
        self,
        tumor_density: torch.Tensor,
        immune_density: torch.Tensor
    ) -> torch.Tensor:
        """
        Entropy regularization based on statistical physics
        
        For a density field ρ(x), the entropy is:
        S = -∫ ρ(x) log(ρ(x)) dx
        
        This encourages smooth, physically plausible distributions
        and prevents unphysical concentration spikes
        """
        eps = 1e-10
        
        # Normalize to get probability distributions
        tumor_norm = tumor_density / (tumor_density.sum(dim=[2, 3], keepdim=True) + eps)
        immune_norm = immune_density / (immune_density.sum(dim=[2, 3], keepdim=True) + eps)
        
        # Compute entropy
        tumor_entropy = -torch.sum(tumor_norm * torch.log(tumor_norm + eps), dim=[2, 3])
        immune_entropy = -torch.sum(immune_norm * torch.log(immune_norm + eps), dim=[2, 3])
        
        # We want high entropy (smooth distributions), so we minimize negative entropy
        entropy_loss = -torch.mean(tumor_entropy + immune_entropy)
        
        return entropy_loss
    
    def forward(
        self,
        tumor_pred: torch.Tensor,
        immune_pred: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss
        
        Returns:
            total_loss: Combined physics loss
            loss_dict: Dictionary of individual loss components
        """
        pde_loss = self.compute_pde_residual(tumor_pred, immune_pred, params)
        
        boundary_loss = self.compute_boundary_loss(tumor_pred) + self.compute_boundary_loss(immune_pred)
        
        entropy_loss = self.compute_entropy_regularization(
            tumor_pred[:, 0:1], immune_pred[:, 0:1]
        )
        
        total_loss = (
            self.pde_weight * pde_loss +
            self.boundary_weight * boundary_loss +
            self.entropy_weight * entropy_loss
        )
        
        loss_dict = {
            'pde_residual': pde_loss.item(),
            'boundary': boundary_loss.item(),
            'entropy': entropy_loss.item()
        }
        
        return total_loss, loss_dict


class PhysicalParameterExtractor(nn.Module):
    """
    Neural network to extract physical parameters from patient data
    
    Takes multimodal features and predicts:
    - Diffusion coefficients (D_tumor, D_immune)
    - Proliferation rates (r_tumor, r_immune)
    - Interaction strengths (α, β)
    - Carrying capacity (K)
    - Initial conditions for density fields
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256, 128],
        num_params: int = 8,
        spatial_resolution: int = 64,
        param_bounds: Optional[dict] = None
    ):
        """
        Args:
            input_dim: Dimension of fused multimodal features
            hidden_dims: Hidden layer dimensions
            num_params: Number of physical parameters to extract
            spatial_resolution: Grid size for density fields
            param_bounds: Dict with min/max bounds for each parameter type
        """
        super().__init__()
        
        self.num_params = num_params
        self.resolution = spatial_resolution
        
        # Parameter bounds for physical interpretability
        self.param_bounds = param_bounds or {
            'diffusion': (0.01, 100.0),
            'proliferation': (0.0, 2.0),
            'interaction': (-10.0, 10.0),
            'carrying_capacity': (0.1, 10.0)
        }
        
        # Network to predict global physical parameters
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.param_network = nn.Sequential(*layers)
        
        # Output heads for different parameter types
        self.diffusion_head = nn.Linear(prev_dim, 2)  # D_tumor, D_immune
        self.proliferation_head = nn.Linear(prev_dim, 2)  # r_tumor, r_immune
        self.interaction_head = nn.Linear(prev_dim, 1)  # α (tumor-immune interaction)
        self.capacity_head = nn.Linear(prev_dim, 1)  # K
        
        # Network to predict initial spatial distributions
        self.spatial_network = nn.Sequential(
            nn.Linear(prev_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * spatial_resolution * spatial_resolution)  # tumor + immune
        )
    
    def forward(self, fused_features: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Extract physical parameters from multimodal features
        
        Args:
            fused_features: [batch, input_dim]
        
        Returns:
            params: Dictionary of physical parameters
            initial_densities: [batch, 2, H, W] initial conditions for tumor and immune cells
        """
        batch_size = fused_features.shape[0]
        
        # Extract parameter features
        param_features = self.param_network(fused_features)
        
        # Predict parameters with appropriate activation functions and bounds
        diffusion_raw = self.diffusion_head(param_features)
        diffusion = self.param_bounds['diffusion'][0] + torch.sigmoid(diffusion_raw) * (
            self.param_bounds['diffusion'][1] - self.param_bounds['diffusion'][0]
        )
        
        proliferation_raw = self.proliferation_head(param_features)
        proliferation = self.param_bounds['proliferation'][0] + torch.sigmoid(proliferation_raw) * (
            self.param_bounds['proliferation'][1] - self.param_bounds['proliferation'][0]
        )
        
        interaction_raw = self.interaction_head(param_features)
        interaction = self.param_bounds['interaction'][0] + torch.sigmoid(interaction_raw) * (
            self.param_bounds['interaction'][1] - self.param_bounds['interaction'][0]
        )
        
        capacity_raw = self.capacity_head(param_features)
        carrying_capacity = self.param_bounds['carrying_capacity'][0] + torch.sigmoid(capacity_raw) * (
            self.param_bounds['carrying_capacity'][1] - self.param_bounds['carrying_capacity'][0]
        )
        
        params = {
            'diffusion_tumor': diffusion[:, 0],
            'diffusion_immune': diffusion[:, 1],
            'proliferation_tumor': proliferation[:, 0],
            'proliferation_immune': proliferation[:, 1],
            'interaction_tumor_immune': interaction[:, 0],
            'carrying_capacity': carrying_capacity[:, 0]
        }
        
        # Predict initial spatial distributions
        spatial_raw = self.spatial_network(param_features)
        spatial_reshaped = spatial_raw.view(batch_size, 2, self.resolution, self.resolution)
        
        # Apply softplus to ensure non-negativity
        initial_densities = torch.nn.functional.softplus(spatial_reshaped)
        
        return params, initial_densities


if __name__ == "__main__":
    # Test physics components
    print("Testing Reaction-Diffusion PDE solver...")
    
    pde = ReactionDiffusionPDE(spatial_resolution=64, dt=0.1)
    
    # Create initial conditions
    tumor_init = torch.randn(4, 1, 64, 64).abs()
    immune_init = torch.randn(4, 1, 64, 64).abs() * 0.5
    
    params = {
        'diffusion_tumor': torch.tensor([1.0, 2.0, 1.5, 0.8]),
        'diffusion_immune': torch.tensor([2.0, 1.5, 2.5, 1.0]),
        'proliferation_tumor': torch.tensor([0.5, 0.6, 0.4, 0.7]),
        'proliferation_immune': torch.tensor([0.3, 0.4, 0.35, 0.3]),
        'interaction_tumor_immune': torch.tensor([0.1, 0.15, 0.12, 0.08]),
        'carrying_capacity': torch.tensor([5.0, 4.5, 5.5, 6.0])
    }
    
    # Simulate
    tumor_traj, immune_traj = pde.simulate_trajectory(
        tumor_init, immune_init, params, num_steps=50, noise_scale=0.01
    )
    
    print(f"Tumor trajectory shape: {tumor_traj.shape}")
    print(f"Immune trajectory shape: {immune_traj.shape}")
    print(f"Final tumor density: {tumor_traj[:, -1].mean():.4f} ± {tumor_traj[:, -1].std():.4f}")
    
    print("\nTesting Physics-Informed Loss...")
    physics_loss = PhysicsInformedLoss(spatial_resolution=64)
    
    tumor_pred = torch.stack([tumor_traj[:, 0], tumor_traj[:, 10]], dim=1)
    immune_pred = torch.stack([immune_traj[:, 0], immune_traj[:, 10]], dim=1)
    
    loss, loss_dict = physics_loss(tumor_pred, immune_pred, params)
    print(f"Total physics loss: {loss.item():.6f}")
    print(f"Loss components: {loss_dict}")
    
    print("\nPhysics components ready!")
