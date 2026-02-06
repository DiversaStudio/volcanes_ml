# %%
# Import libraries
import os
import sys
import torch
import gc
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# %%
# Set path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# %%
# Import modules
from src.data.dataset import ThermalDataset
from src.utils.visualization import plot_class_distribution
from src.data.dataset import save_large_dataset

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# Load preprocessed dataset
print("\nLoading preprocessed dataset...")
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
preprocessed_dataset_path = os.path.join(project_root, 'data', 'processed', 'preprocessed_dataset.pt')
preprocessed_dataset = torch.load(preprocessed_dataset_path, weights_only=False)

# %%
# Print shapes before any operations
print("\nOriginal tensor shapes:")
print(f"Corrected: {preprocessed_dataset['tensors']['corrected'].shape}")

# %%
# We need to permute the tensors before splitting
print("\nReorganizing tensors...")
preprocessed_dataset['tensors']['corrected'] = preprocessed_dataset['tensors']['corrected'].permute(2, 0, 1)
preprocessed_dataset['tensors']['edge'] = preprocessed_dataset['tensors']['edge'].permute(2, 0, 1)
for level in preprocessed_dataset['tensors']['threshold'].keys():
    preprocessed_dataset['tensors']['threshold'][level] = preprocessed_dataset['tensors']['threshold'][level].permute(2, 0, 1)

print("\nTensor shapes after permute:")
print(f"Corrected: {preprocessed_dataset['tensors']['corrected'].shape}")

# %%
# Now split using the correct dimension
n_samples = len(preprocessed_dataset['labels']['numeric_labels'])
print(f"\nTotal number of samples: {n_samples}")

# Split data
print("\nSplitting data into train and validation sets...")
labels = preprocessed_dataset['labels']['numeric_labels']
train_idx, val_idx = train_test_split(
    range(n_samples), 
    test_size=0.2, 
    stratify=labels,
    random_state=42
)

print(f"Training indices: {len(train_idx)}, max index: {max(train_idx)}")
print(f"Validation indices: {len(val_idx)}, max index: {max(val_idx)}")

# %%
# Create datasets
train_dataset = ThermalDataset(preprocessed_dataset, indices=train_idx)
val_dataset = ThermalDataset(preprocessed_dataset, indices=val_idx)

# %%
# Print final dataset information
print("\nFinal dataset splits:")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# %%
# Create processed directory
processed_dir = os.path.join(project_root, 'data', 'processed')
os.makedirs(processed_dir, exist_ok=True)

# Save datasets
print("\nSaving datasets...")
save_large_dataset(train_dataset, processed_dir, "train")
save_large_dataset(val_dataset, processed_dir, "val")

# %%
gc.collect()
torch.cuda.empty_cache()

# %%
plot_class_distribution(train_dataset, val_dataset)

# %%



