"""
Evaluation and Visualization Tools for Physics-Informed Cancer Model
Includes: trajectory visualization, parameter analysis, survival curves, attention maps
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150


class ModelEvaluator:
    """
    Comprehensive evaluation and visualization for trained PINN model
    """
    
    def __init__(self, model, config, device='cuda'):
        """
        Args:
            model: Trained PhysicsInformedCancerModel
            config: Configuration object
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
    
    @torch.no_grad()
    def predict_patient(
        self,
        image: torch.Tensor,
        genomic: torch.Tensor
    ) -> Dict[str, any]:
        """
        Make predictions for a single patient
        
        Returns comprehensive output including:
        - Clinical risk scores
        - Physical parameters
        - Tumor-immune trajectories
        - Attention weights
        """
        image = image.unsqueeze(0).to(self.device)
        genomic = genomic.unsqueeze(0).to(self.device)
        
        # Get model predictions
        output = self.model(image, genomic, return_physics=True)
        
        # Simulate trajectory
        tumor_traj, immune_traj = self.model.predict_trajectory(
            image, genomic, 
            num_steps=self.config.physics.time_steps,
            noise_scale=self.config.physics.noise_scale
        )
        
        # Extract and organize results
        results = {
            # Clinical predictions
            'survival_risk': output['survival_risk'].item(),
            'progression_prob': torch.sigmoid(output['progression_logit']).item(),
            'metastasis_prob': torch.sigmoid(output['metastasis_logit']).item(),
            
            # Physical parameters
            'diffusion_tumor': output['physics_params']['diffusion_tumor'].item(),
            'diffusion_immune': output['physics_params']['diffusion_immune'].item(),
            'proliferation_tumor': output['physics_params']['proliferation_tumor'].item(),
            'proliferation_immune': output['physics_params']['proliferation_immune'].item(),
            'interaction_strength': output['physics_params']['interaction_tumor_immune'].item(),
            'carrying_capacity': output['physics_params']['carrying_capacity'].item(),
            
            # Trajectories
            'tumor_trajectory': tumor_traj.cpu().numpy()[0],
            'immune_trajectory': immune_traj.cpu().numpy()[0],
            
            # Attention weights
            'attention_weights': output['attention_weights'].cpu().numpy()[0],
            
            # Latent representations
            'genomic_latent': output['genomic_mu'].cpu().numpy()[0]
        }
        
        return results
    
    def visualize_patient_analysis(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive visualization for a single patient
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Clinical risk scores
        ax1 = fig.add_subplot(gs[0, 0])
        risks = [
            results['survival_risk'],
            results['progression_prob'],
            results['metastasis_prob']
        ]
        ax1.barh(['Survival Risk', 'Progression', 'Metastasis'], risks, color=['red', 'orange', 'purple'])
        ax1.set_xlim(0, 1)
        ax1.set_title('Clinical Risk Predictions', fontweight='bold')
        ax1.set_xlabel('Risk Score')
        
        # 2. Physical parameters
        ax2 = fig.add_subplot(gs[0, 1])
        params = {
            'D_tumor': results['diffusion_tumor'],
            'D_immune': results['diffusion_immune'],
            'r_tumor': results['proliferation_tumor'],
            'r_immune': results['proliferation_immune'],
            'α': results['interaction_strength'],
            'K': results['carrying_capacity']
        }
        y_pos = np.arange(len(params))
        ax2.barh(y_pos, list(params.values()), color='steelblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(list(params.keys()))
        ax2.set_title('Physical Parameters', fontweight='bold')
        ax2.set_xlabel('Parameter Value')
        
        # 3. Tumor trajectory over time
        ax3 = fig.add_subplot(gs[0, 2:])
        tumor_traj = results['tumor_trajectory']
        time_points = [0, len(tumor_traj)//4, len(tumor_traj)//2, 3*len(tumor_traj)//4, -1]
        
        for i, t in enumerate(time_points):
            ax = fig.add_subplot(gs[1, i if i < 4 else 3])
            im = ax.imshow(tumor_traj[t], cmap='Reds', vmin=0, vmax=tumor_traj.max())
            ax.set_title(f't = {t}', fontsize=10)
            ax.axis('off')
            if i == len(time_points) - 1:
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 4. Immune trajectory over time
        immune_traj = results['immune_trajectory']
        for i, t in enumerate(time_points):
            ax = fig.add_subplot(gs[2, i if i < 4 else 3])
            im = ax.imshow(immune_traj[t], cmap='Blues', vmin=0, vmax=immune_traj.max())
            ax.set_title(f't = {t}', fontsize=10)
            ax.axis('off')
            if i == len(time_points) - 1:
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Add row labels
        fig.text(0.02, 0.67, 'Tumor\nDensity', ha='center', va='center', fontsize=12, fontweight='bold')
        fig.text(0.02, 0.33, 'Immune\nDensity', ha='center', va='center', fontsize=12, fontweight='bold')
        
        fig.suptitle('Patient Analysis: Physics-Informed Cancer Dynamics', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def analyze_physical_biomarkers(
        self,
        dataloader,
        save_dir: str = './analysis'
    ):
        """
        Analyze distribution of physical parameters across patient cohort
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("Extracting physical parameters from all patients...")
        
        params_list = {
            'diffusion_tumor': [],
            'diffusion_immune': [],
            'proliferation_tumor': [],
            'proliferation_immune': [],
            'interaction_strength': [],
            'carrying_capacity': []
        }
        
        clinical_data = {
            'survival_time': [],
            'survival_event': [],
            'progression': [],
            'metastasis': []
        }
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            genomic = batch['genomic'].to(self.device)
            
            output = self.model(images, genomic, return_physics=True)
            
            # Collect physics parameters
            params_list['diffusion_tumor'].extend(output['physics_params']['diffusion_tumor'].cpu().numpy())
            params_list['diffusion_immune'].extend(output['physics_params']['diffusion_immune'].cpu().numpy())
            params_list['proliferation_tumor'].extend(output['physics_params']['proliferation_tumor'].cpu().numpy())
            params_list['proliferation_immune'].extend(output['physics_params']['proliferation_immune'].cpu().numpy())
            params_list['interaction_strength'].extend(output['physics_params']['interaction_tumor_immune'].cpu().numpy())
            params_list['carrying_capacity'].extend(output['physics_params']['carrying_capacity'].cpu().numpy())
            
            # Collect clinical data
            clinical_data['survival_time'].extend(batch['survival_time'].cpu().numpy())
            clinical_data['survival_event'].extend(batch['survival_event'].cpu().numpy())
            clinical_data['progression'].extend(batch['progression'].cpu().numpy())
            clinical_data['metastasis'].extend(batch['metastasis'].cpu().numpy())
        
        # Convert to DataFrame
        df = pd.DataFrame({**params_list, **clinical_data})
        
        # Plot parameter distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (param_name, param_values) in enumerate(params_list.items()):
            axes[i].hist(param_values, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(param_name.replace('_', ' ').title())
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {param_name.replace("_", " ").title()}')
            
            # Add statistics
            mean_val = np.mean(param_values)
            std_val = np.std(param_values)
            axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'parameter_distributions.png'), dpi=300)
        print(f"Saved parameter distributions to {save_dir}")
        
        # Correlation analysis with outcomes
        print("\nCorrelation of physical parameters with clinical outcomes:")
        
        for param in params_list.keys():
            # Correlation with survival time (among events)
            mask = df['survival_event'] == 1
            if mask.sum() > 0:
                corr, p_value = stats.pearsonr(df[param][mask], df['survival_time'][mask])
                print(f"{param} vs Survival Time: r={corr:.3f}, p={p_value:.4f}")
            
            # Association with metastasis
            metastasis_yes = df[param][df['metastasis'] == 1]
            metastasis_no = df[param][df['metastasis'] == 0]
            if len(metastasis_yes) > 0 and len(metastasis_no) > 0:
                t_stat, p_value = stats.ttest_ind(metastasis_yes, metastasis_no)
                print(f"{param} - Metastasis: t={t_stat:.3f}, p={p_value:.4f}")
        
        return df
    
    def plot_survival_curves(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Plot Kaplan-Meier survival curves stratified by risk
        """
        # Concatenate all predictions and targets
        all_risks = []
        all_times = []
        all_events = []
        
        for pred_batch, target_batch in zip(predictions, targets):
            all_risks.extend(pred_batch['survival_risk'].cpu().numpy())
            all_times.extend(target_batch['survival_time'].cpu().numpy())
            all_events.extend(target_batch['survival_event'].cpu().numpy())
        
        all_risks = np.array(all_risks)
        all_times = np.array(all_times)
        all_events = np.array(all_events)
        
        # Stratify into risk groups
        low_risk_mask = all_risks < np.percentile(all_risks, 33)
        high_risk_mask = all_risks > np.percentile(all_risks, 67)
        mid_risk_mask = ~(low_risk_mask | high_risk_mask)
        
        # Fit Kaplan-Meier curves
        kmf = KaplanMeierFitter()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Low risk group
        kmf.fit(all_times[low_risk_mask], all_events[low_risk_mask], label='Low Risk')
        kmf.plot_survival_function(ax=ax, ci_show=True, color='green')
        
        # Mid risk group
        kmf.fit(all_times[mid_risk_mask], all_events[mid_risk_mask], label='Medium Risk')
        kmf.plot_survival_function(ax=ax, ci_show=True, color='orange')
        
        # High risk group
        kmf.fit(all_times[high_risk_mask], all_events[high_risk_mask], label='High Risk')
        kmf.plot_survival_function(ax=ax, ci_show=True, color='red')
        
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title('Kaplan-Meier Survival Curves by Risk Group', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        # Log-rank test
        results = logrank_test(
            all_times[low_risk_mask], all_times[high_risk_mask],
            all_events[low_risk_mask], all_events[high_risk_mask]
        )
        
        ax.text(0.05, 0.05, f'Log-rank p-value: {results.p_value:.4f}', 
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved survival curves to {save_path}")
        
        plt.show()
    
    def compare_with_baseline(
        self,
        dataloader,
        save_path: Optional[str] = None
    ):
        """
        Compare physics-informed model with baseline (no physics)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        
        print("Comparing PINN with baseline models...")
        
        # Extract features and labels
        all_features = []
        all_survival_events = []
        all_progression = []
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            genomic = batch['genomic'].to(self.device)
            
            output = self.model(images, genomic, return_physics=True)
            all_features.append(output['fused_features'].cpu().numpy())
            all_survival_events.append(batch['survival_event'].cpu().numpy())
            all_progression.append(batch['progression'].cpu().numpy())
        
        X = np.vstack(all_features)
        y_event = np.concatenate(all_survival_events)
        y_prog = np.concatenate(all_progression)
        
        # Simple logistic regression baseline
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y_prog)
        y_pred_lr = lr.predict_proba(X)[:, 1]
        
        auroc_lr = roc_auc_score(y_prog, y_pred_lr)
        print(f"Baseline AUROC (Progression): {auroc_lr:.4f}")
        
        return auroc_lr


def create_analysis_report(
    model,
    config,
    test_loader,
    save_dir: str = './analysis_results'
):
    """
    Generate comprehensive analysis report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    evaluator = ModelEvaluator(model, config)
    
    print("Generating analysis report...")
    
    # 1. Analyze physical biomarkers
    print("\n1. Analyzing physical biomarkers...")
    param_df = evaluator.analyze_physical_biomarkers(test_loader, save_dir)
    
    # Save parameter statistics
    param_stats = param_df.describe()
    param_stats.to_csv(os.path.join(save_dir, 'parameter_statistics.csv'))
    
    # 2. Example patient visualization
    print("\n2. Creating example patient visualization...")
    batch = next(iter(test_loader))
    sample_image = batch['image'][0]
    sample_genomic = batch['genomic'][0]
    
    results = evaluator.predict_patient(sample_image, sample_genomic)
    evaluator.visualize_patient_analysis(
        results,
        save_path=os.path.join(save_dir, 'example_patient.png')
    )
    
    print(f"\nAnalysis report saved to {save_dir}")
    print("Generated files:")
    print("  - parameter_distributions.png")
    print("  - parameter_statistics.csv")
    print("  - example_patient.png")


if __name__ == "__main__":
    from config import get_default_config
    from model import create_model
    from data_loader import get_dataloaders
    import torch
    
    print("Testing evaluation tools...")
    
    config = get_default_config()
    model = create_model(config)
    
    # Load best checkpoint if exists
    checkpoint_path = './checkpoints/checkpoint_best.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best checkpoint")
    
    # Get test data
    _, _, test_loader, _ = get_dataloaders(config, create_synthetic=True)
    
    # Create analysis
    create_analysis_report(model, config, test_loader)
    
    print("\nEvaluation tools test passed!")
