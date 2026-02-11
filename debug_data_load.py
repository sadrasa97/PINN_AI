from config import get_default_config
from data_loader import get_dataloaders

config = get_default_config()
print(f"Config gene expression dim: {config.data.gene_expression_dim}")

train_loader, val_loader, test_loader, scaler = get_dataloaders(config, use_real_data=True)

batch = next(iter(train_loader))
print(f'Genomic shape: {batch["genomic"].shape}')
print(f'Expected shape: ({config.data.batch_size}, {config.data.gene_expression_dim})')

# Also check the actual data file
import pandas as pd
df = pd.read_csv('./data/transcriptomics.csv', index_col=0)
print(f'Data file shape: {df.shape}')