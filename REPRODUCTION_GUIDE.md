# Reproduction Guide: Physics-Informed Neural Network for Cancer Dynamics

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Complete Workflow](#complete-workflow)
5. [Validation Checklist](#validation-checklist)
6. [Expected Results](#expected-results)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **CPU**: Intel i5 or AMD Ryzen 5 (4+ cores)
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 2080, Tesla T4)
- **Storage**: 20 GB free space
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows 10+

### Recommended Requirements
- **CPU**: Intel i7/i9 or AMD Ryzen 7/9 (8+ cores)
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU with 16GB+ VRAM (e.g., RTX 3090, A100)
- **Storage**: 50 GB SSD
- **OS**: Linux (Ubuntu 22.04)

### Software Dependencies
- Python 3.8 - 3.11
- CUDA 11.7+ (for GPU acceleration)
- cuDNN 8.0+

---

## Installation

### Step 1: Set Up Python Environment

```bash
# Create virtual environment
python -m venv pinn_env

# Activate environment
# On Linux/Mac:
source pinn_env/bin/activate
# On Windows:
pinn_env\Scripts\activate

# Verify Python version
python --version  # Should be 3.8+
```

### Step 2: Install PyTorch

Visit [pytorch.org](https://pytorch.org) and install the appropriate version for your system.

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchvision
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.1.0+cu118
CUDA available: True
```

---

## Quick Start

### Run Integration Tests

Verify all components work correctly:

```bash
python test_integration.py
```

Expected output:
```
================================================================================
INTEGRATION TEST SUITE
================================================================================

1. Testing imports...
✓ All imports successful

2. Testing configuration...
✓ Configuration valid
  Device: cuda
  Batch size: 8

[... additional tests ...]

================================================================================
TEST SUMMARY
================================================================================
✓ PASS: Imports
✓ PASS: Configuration
✓ PASS: Data Pipeline
✓ PASS: Physics Components
✓ PASS: Model Architecture
✓ PASS: Loss Functions
✓ PASS: Training Step
✓ PASS: Evaluation

Result: 8/8 tests passed

🎉 All tests passed! System is ready for use.
```

### Quick Experiment (5-10 minutes)

Run a minimal training experiment to verify the system:

```bash
python main.py --mode full --num_epochs 5 --batch_size 4
```

This will:
1. Generate 500 synthetic patient samples
2. Train for 5 epochs (~5-10 minutes on GPU)
3. Evaluate on test set
4. Generate analysis report

---

## Complete Workflow

### Step 1: Generate Dataset

If using synthetic data (for testing):

```bash
python -c "from data_loader import create_synthetic_tcga_data; \
create_synthetic_tcga_data(num_samples=500, output_dir='./data')"
```

If using real TCGA data:
- Download TCGA data from [GDC Data Portal](https://portal.gdc.cancer.gov/)
- Organize as:
  ```
  data/
  ├── histopathology/
  │   ├── TCGA-0001.png
  │   └── ...
  ├── transcriptomics.csv  (genes × samples)
  └── clinical_outcomes.csv
  ```

### Step 2: Configure Experiment

Edit `config.py` or create custom config:

```python
from config import Config

config = Config()

# Adjust hyperparameters
config.training.learning_rate = 1e-4
config.training.num_epochs = 200
config.training.early_stopping_patience = 30

# Adjust model architecture
config.model.image_encoder_type = "resnet34"
config.model.genomic_latent_dim = 256

# Adjust physics parameters
config.physics.spatial_resolution = 64
config.physics.pde_residual_weight = 1.0
```

### Step 3: Train Model

**Option A: Using main script**
```bash
python main.py --mode train --num_epochs 200 --batch_size 8
```

**Option B: Using train.py directly**
```python
from config import get_default_config
from train import Trainer

config = get_default_config()
trainer = Trainer(config)
trainer.train()
```

Training will:
- Save checkpoints to `./checkpoints/`
- Log metrics to TensorBoard in `./runs/`
- Display progress with tqdm bars
- Perform validation every epoch
- Apply early stopping if validation loss plateaus

### Step 4: Monitor Training

Open TensorBoard to visualize training:

```bash
tensorboard --logdir ./runs --port 6006
```

Navigate to `http://localhost:6006` in your browser.

**Key metrics to monitor:**
- `train/total`: Total loss (should decrease)
- `train/c_index`: Concordance index (should increase toward 1.0)
- `val/total`: Validation loss (check for overfitting)
- `physics/pde_residual`: PDE constraint violation (should be small)

### Step 5: Evaluate Model

After training, evaluate on test set:

```bash
python main.py --mode eval --checkpoint ./checkpoints/checkpoint_best.pt
```

Or programmatically:

```python
from train import Trainer
from config import get_default_config

config = get_default_config()
trainer = Trainer(config)
results, predictions, targets = trainer.test('./checkpoints/checkpoint_best.pt')

print(f"C-index: {results['c_index']:.4f}")
print(f"Progression AUROC: {results['progression_auroc']:.4f}")
```

### Step 6: Generate Analysis Report

Create comprehensive visualizations and analysis:

```bash
python main.py --mode eval --generate_report --output_dir ./analysis_results
```

This produces:
- `parameter_distributions.png`: Physical parameter histograms
- `parameter_statistics.csv`: Descriptive statistics
- `example_patient.png`: Single patient analysis with trajectories
- `test_results.json`: Numerical results

---

## Validation Checklist

### ✓ Data Pipeline Validation

- [ ] Dataset loads without errors
- [ ] Batch shapes are correct:
  - Images: `[batch, 3, 512, 512]`
  - Genomic: `[batch, 20530]`
  - Labels present: survival_time, survival_event, progression, metastasis
- [ ] Color normalization applied (if using Macenko)
- [ ] No NaN or infinite values in data

```python
from data_loader import get_dataloaders
from config import get_default_config

config = get_default_config()
train_loader, _, _, _ = get_dataloaders(config, create_synthetic=True)

batch = next(iter(train_loader))
assert not torch.isnan(batch['image']).any()
assert not torch.isnan(batch['genomic']).any()
print("✓ Data validation passed")
```

### ✓ Model Architecture Validation

- [ ] Model creates successfully
- [ ] Forward pass completes
- [ ] Output shapes correct
- [ ] Gradients flow through all components
- [ ] Physics parameters in valid ranges

```python
from model import create_model
from config import get_default_config

config = get_default_config()
model = create_model(config)

# Test forward
images = torch.randn(2, 3, 512, 512)
genomic = torch.randn(2, 20530)
output = model(images, genomic, return_physics=True)

assert 'survival_risk' in output
assert 'physics_params' in output
assert output['physics_params']['diffusion_tumor'].min() >= 0.01
print("✓ Model validation passed")
```

### ✓ Physics Validation

- [ ] PDE solver runs without errors
- [ ] Solutions remain non-negative (density constraint)
- [ ] Boundary conditions satisfied
- [ ] Parameters within physical bounds
- [ ] Energy/entropy calculations stable

```python
from physics_informed import ReactionDiffusionPDE

pde = ReactionDiffusionPDE(spatial_resolution=64)
tumor = torch.rand(1, 1, 64, 64)
immune = torch.rand(1, 1, 64, 64) * 0.5

params = {
    'diffusion_tumor': torch.tensor([1.0]),
    'diffusion_immune': torch.tensor([2.0]),
    'proliferation_tumor': torch.tensor([0.5]),
    'proliferation_immune': torch.tensor([0.3]),
    'interaction_tumor_immune': torch.tensor([0.1]),
    'carrying_capacity': torch.tensor([5.0])
}

tumor_next, immune_next = pde.step(tumor, immune, params)
assert (tumor_next >= 0).all(), "Densities must be non-negative"
print("✓ Physics validation passed")
```

### ✓ Training Validation

- [ ] Loss decreases over epochs
- [ ] Validation metrics improve
- [ ] No gradient explosions (gradients < 100)
- [ ] Checkpoints save correctly
- [ ] TensorBoard logs properly

### ✓ Evaluation Validation

- [ ] Test metrics computed correctly
- [ ] C-index between 0.5 and 1.0
- [ ] AUROC between 0.5 and 1.0
- [ ] Physical parameters statistically reasonable
- [ ] Visualizations render without errors

---

## Expected Results

### Synthetic Data Performance

When training on synthetic data (500 samples, 200 epochs):

**Clinical Metrics:**
```
Test Set Results:
├── C-index (Survival): 0.68 - 0.75
├── Progression AUROC: 0.72 - 0.82
├── Progression Accuracy: 0.65 - 0.78
├── Metastasis AUROC: 0.70 - 0.80
└── Metastasis Accuracy: 0.63 - 0.75
```

**Physics Metrics:**
```
Physical Parameters:
├── Diffusion coefficients: 0.1 - 50.0 μm²/day
├── Proliferation rates: 0.0 - 1.5 day⁻¹
├── Interaction strengths: -5.0 - 5.0
├── PDE residual: < 0.01
└── Boundary loss: < 0.005
```

**Training Time:**
- GPU (RTX 3090): ~20-30 minutes for 200 epochs
- GPU (Tesla T4): ~45-60 minutes for 200 epochs
- CPU: ~4-6 hours for 200 epochs (not recommended)

### Performance Indicators

**Good Training:**
- Loss decreases smoothly
- Validation tracks training (small gap)
- C-index > 0.65
- AUROC > 0.70
- PDE residual < 0.01

**Warning Signs:**
- Loss oscillates wildly → Reduce learning rate
- Validation diverges from training → Reduce model complexity or increase regularization
- C-index < 0.55 → Check data quality or increase epochs
- PDE residual > 0.1 → Increase physics loss weight

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size
config.data.batch_size = 4  # or even 2

# Reduce spatial resolution
config.physics.spatial_resolution = 32

# Use gradient checkpointing
torch.utils.checkpoint.checkpoint(...)

# Enable mixed precision
config.training.mixed_precision = True
```

### Issue 2: Slow Training

**Symptoms:**
- Training takes hours per epoch
- GPU utilization < 50%

**Solutions:**
```python
# Increase number of data loading workers
config.data.num_workers = 8

# Enable mixed precision
config.training.mixed_precision = True

# Reduce physics computation frequency
# Compute physics loss every N batches instead of every batch

# Use smaller image size
config.data.image_size = (256, 256)
```

### Issue 3: Poor Convergence

**Symptoms:**
- Loss plateaus early
- Metrics don't improve

**Solutions:**
```python
# Adjust loss weights
config.training.clinical_loss_weight = 1.0
config.training.physics_loss_weight = 0.1  # Start small

# Try different learning rate
config.training.learning_rate = 5e-5

# Increase model capacity
config.model.image_output_dim = 1024
config.model.fusion_dim = 1024

# More epochs
config.training.num_epochs = 300
```

### Issue 4: NaN Loss

**Symptoms:**
```
loss = nan
```

**Solutions:**
```python
# Reduce learning rate
config.training.learning_rate = 1e-5

# Gradient clipping
config.training.gradient_clip_norm = 0.5

# Check for numerical stability
# - Ensure inputs are normalized
# - Add epsilon to log/sqrt operations
# - Use torch.nn.functional.softplus instead of exp

# Initialize parameters carefully
torch.nn.init.xavier_normal_(...)
```

### Issue 5: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'xyz'
```

**Solutions:**
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.8+

# Check virtual environment is activated
which python  # Should point to venv

# Install missing packages individually
pip install lifelines matplotlib seaborn
```

---

## Contact & Support

For additional help:

1. **Check test suite**: `python test_integration.py`
2. **Review logs**: Check `./runs/` for TensorBoard logs
3. **Examine checkpoints**: Ensure models save correctly
4. **GitHub Issues**: Report bugs with full error logs

---

## Reproducibility Checklist

Before publishing results, ensure:

- [ ] All random seeds set (`config.training.seed = 42`)
- [ ] Deterministic mode enabled (`config.training.deterministic = True`)
- [ ] Hardware specifications documented
- [ ] Software versions recorded (`pip freeze > environment.txt`)
- [ ] Checkpoints saved and backed up
- [ ] Results logged to JSON files
- [ ] Visualizations saved with high DPI (300+)
- [ ] Statistical tests performed (bootstrap, confidence intervals)
- [ ] Code version controlled (git commit hash recorded)

---

## Citation

When publishing results, include:

```
Model Implementation:
- Framework: PyTorch 2.1.0
- CUDA: 11.8
- Random Seed: 42
- Training Time: XX hours on NVIDIA YYY
- Code Version: [git commit hash]
```

---

**Last Updated**: 2024
**Framework Version**: 1.0.0
