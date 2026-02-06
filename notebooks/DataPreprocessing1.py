# %% [markdown]
# ## Setup

# %% [markdown]
# ### Import Libraries

# %%
import os
import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# %% [markdown]
# ### Import modules

# %%
# Set path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# %%
# Import modules
from src.data.loader import ThermalDataLoader
from src.features.thermal import create_thermal_threshold_tensor
from src.features.edge_detection import create_edge_detection_tensors
from src.data.preprocessor import prepare_labels
from src.utils.visualization import (
    get_thermal_stats,
    print_thermal_stats,
    visualize_thermal_sequence,
    get_edge_stats,
    print_edge_stats,
    visualize_edge_sequence,
    visualize_thermal_threshold_comparison,
    get_label_examples
)

# %% [markdown]
# ### Set CUDA

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Functions

# %%
def preprocess_dataset(dataset, thresholds=None):
    """
    Process any dataset by converting to tensors, creating features, and organizing data.
    
    Args:
        dataset: Dictionary containing the raw dataset with 'tensors', 'metadata', etc.
        
    Returns:
        dict: Preprocessed dataset with organized tensors, metadata, and labels
    """
    if dataset is None:
        raise ValueError("Dataset cannot be None")
    
    print("Starting dataset preprocessing...")
    
    # Convert numpy arrays to PyTorch tensors and check for NaN
    print("\nConverting tensors...")
    corrected_tensor = torch.tensor(dataset['tensors']['corrected']).float()
    
    # Check for NaN values in corrected tensor
    if torch.isnan(corrected_tensor).any():
        print("Warning: NaN values found in corrected tensor!")
        print(f"Number of NaN values: {torch.isnan(corrected_tensor).sum().item()}")
        # Replace NaN with 0
        corrected_tensor = torch.nan_to_num(corrected_tensor, nan=0.0)
    
    # Create all initial tensors
    print("\nCreating feature tensors...")
    
    # Create edge detection tensors
    print("- Creating edge detection tensors...")
    edge_detection_data = create_edge_detection_tensors(corrected_tensor)
    edge_tensor = torch.tensor(edge_detection_data['edge_tensor']).float()
    edge_features = torch.tensor(edge_detection_data['edge_features']).float()
    
    # Check edge tensors for NaN
    if torch.isnan(edge_tensor).any():
        print("Warning: NaN values found in edge tensor!")
        edge_tensor = torch.nan_to_num(edge_tensor, nan=0.0)
    
    if torch.isnan(edge_features).any():
        print("Warning: NaN values found in edge features!")
        edge_features = torch.nan_to_num(edge_features, nan=0.0)
    
    # Create threshold tensors
    print("- Creating threshold tensors...")
    threshold_data = create_thermal_threshold_tensor(corrected_tensor, thresholds=thresholds)
    
    # Check threshold tensors for NaN
    for name, tensor in threshold_data['tensors'].items():
        if torch.isnan(tensor).any():
            print(f"Warning: NaN values found in threshold tensor {name}!")
            threshold_data['tensors'][name] = torch.nan_to_num(tensor, nan=0.0)
    
    # Prepare labels
    print("\nPreparing labels...")
    labels_data = prepare_labels(dataset)
    
    # Create preprocessed dataset dictionary
    print("\nOrganizing preprocessed dataset...")
    preprocessed_dataset = {
        'tensors': {
            'corrected': corrected_tensor,  
            'edge': edge_tensor,
            'edge_features': edge_features,
            'threshold': threshold_data['tensors']
        },
        'metadata': dataset['metadata'],
        'labels': {
            'numeric_labels': labels_data['numeric_labels'],
            'label_mapping': labels_data['label_mapping']
        }
    }
    
    # Print shapes of processed tensors
    print("\nProcessed tensor shapes:")
    print(f"Corrected: {preprocessed_dataset['tensors']['corrected'].shape}")
    print(f"Edge: {preprocessed_dataset['tensors']['edge'].shape}")
    print(f"Edge features: {preprocessed_dataset['tensors']['edge_features'].shape}")
    print("Threshold tensors:")
    for name, tensor in preprocessed_dataset['tensors']['threshold'].items():
        print(f"- {name}: {tensor.shape}")
    print(f"Labels: {preprocessed_dataset['labels']['numeric_labels'].shape}")
    
    print("\nDataset preprocessing completed!")
    return preprocessed_dataset

# %%
def extract_visualization_data(preprocessed_dataset):
    """
    Extract necessary data from preprocessed dataset for visualization.
    
    Args:
        preprocessed_dataset: Output from preprocess_dataset() function
        
    Returns:
        dict: Dictionary containing all data needed for visualization
    """
    return {
        'timestamps': preprocessed_dataset['metadata']['timestamps'],
        'numeric_labels': preprocessed_dataset['labels']['numeric_labels'],
        'label_mapping': preprocessed_dataset['labels']['label_mapping'],
        'corrected_tensor': preprocessed_dataset['tensors']['corrected']
    }

# %%
def resize_tensor(tensor, target_height, target_width, mode='bilinear'):
    """
    Resize a 3D tensor (H, W, T) to (target_height, target_width, T).
    Use mode='nearest' for binary/mask tensors, 'bilinear' for continuous.
    """
    tensor = tensor.permute(2, 0, 1).unsqueeze(1)  # (T, 1, H, W)
    tensor_resized = F.interpolate(
        tensor, size=(target_height, target_width), mode=mode, align_corners=False if mode=='bilinear' else None
    )
    tensor_resized = tensor_resized.squeeze(1).permute(1, 2, 0)  # (H, W, T)
    return tensor_resized

def resize_preprocessed_dataset(preprocessed_dataset, target_height, target_width):
    """
    Resize all spatial tensors in a preprocessed dataset to the target size.
    """
    # Resize corrected tensor
    corrected = preprocessed_dataset['tensors']['corrected']
    preprocessed_dataset['tensors']['corrected'] = resize_tensor(corrected, target_height, target_width, mode='bilinear')

    # Resize edge tensor
    edge = preprocessed_dataset['tensors']['edge']
    preprocessed_dataset['tensors']['edge'] = resize_tensor(edge, target_height, target_width, mode='bilinear')

    # Resize threshold tensors (use nearest for binary masks)
    for name, tensor in preprocessed_dataset['tensors']['threshold'].items():
        preprocessed_dataset['tensors']['threshold'][name] = resize_tensor(tensor, target_height, target_width, mode='nearest')

    return preprocessed_dataset

# %%
def save_preprocessed_dataset(preprocessed_dataset, filename='preprocessed_dataset.pt'):
    """
    Save the preprocessed dataset to the processed directory and print tensor shapes.
    
    Args:
        preprocessed_dataset (dict): The dataset to save.
        filename (str): Name of the file to save.
    """
    # Get project root path (up one level from notebooks)
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    
    # Create processed directory in the correct location
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save the preprocessed dataset
    print("\nSaving preprocessed dataset...")
    preprocessed_dataset_path = os.path.join(processed_dir, filename)
    torch.save(preprocessed_dataset, preprocessed_dataset_path)
    print("Dataset saved successfully!")

    # Print shapes of saved tensors
    print("\nTensor shapes:")
    print(f"Corrected: {preprocessed_dataset['tensors']['corrected'].shape}")
    print(f"Edge: {preprocessed_dataset['tensors']['edge'].shape}")
    print("Threshold tensors:")
    for name, tensor in preprocessed_dataset['tensors']['threshold'].items():
        print(f"- {name}: {tensor.shape}")
    print(f"Labels: {preprocessed_dataset['labels']['numeric_labels'].shape}")

# %% [markdown]
# ## Cotopaxi Tensors

# %% [markdown]
# ### Data Loader

# %%
allowed_labels = {'Nublado', 'Emisiones', 'Despejado', 'Flujo'}

# Create absolute paths
base_directories_COTOPAXI = [
    os.path.join(project_root, 'data/input/Cotopaxi/Nublado'),
    os.path.join(project_root, 'data/input/Cotopaxi/Emisiones'),
    os.path.join(project_root, 'data/input/Cotopaxi/Despejado')
]

# Initialize data loader
print("\nInitializing data loader...")
thermal_loader_cotopaxi = ThermalDataLoader(
    base_directories=base_directories_COTOPAXI,
    limit_per_directory=4500,
    allowed_labels=allowed_labels
)

# %% [markdown]
# ### Create dataset

# %%
# Create initial dataset
print("\nCreating Cotopaxi dataset...")
dataset_cotopaxi = thermal_loader_cotopaxi.create_dataset()

# %%
cotopaxi_thresholds = {'low': 0.1, 'medium': 2.0, 'high': 5.0, 'very_high': 10.00}
preprocessed_cotopaxi = preprocess_dataset(dataset_cotopaxi, thresholds=cotopaxi_thresholds)

# %%
# Resize tensors to target dimensions
target_height, target_width = 240, 320
preprocessed_cotopaxi = resize_preprocessed_dataset(preprocessed_cotopaxi, target_height, target_width)

# %%
save_preprocessed_dataset(preprocessed_cotopaxi, filename='cotopaxi_preprocessed.pt')

# %% [markdown]
# ### Data Visualization

# %% [markdown]
# #### Thermal Sequence

# %%
# Extract data for visualization
viz_data_cotopaxi = extract_visualization_data(preprocessed_cotopaxi)

# Use the extracted data (replaces your original lines)
timestamps = viz_data_cotopaxi['timestamps']
numeric_labels = viz_data_cotopaxi['numeric_labels']
label_mapping = viz_data_cotopaxi['label_mapping']
corrected_tensor = viz_data_cotopaxi['corrected_tensor']

label_examples = get_label_examples(numeric_labels, label_mapping)
example_indices = list(label_examples.values())

thermal_stats = get_thermal_stats(corrected_tensor)
print_thermal_stats(thermal_stats)
visualize_thermal_sequence(
    corrected_tensor,
    timestamps,
    numeric_labels,
    example_indices,
    thermal_stats['global_min'],
    thermal_stats['global_max'],
    label_mapping
)

# %% [markdown]
# #### Thresholds Sequence

# %%
# Thermal threshold visualization
visualize_thermal_threshold_comparison(
    corrected_tensor,
    timestamps,
    numeric_labels,
    example_indices,  # Using the same example_indices from before
    label_mapping,
    thresholds=cotopaxi_thresholds
)

# %% [markdown]
# #### Edges Sequence

# %%
# Edge detection visualization and stats
edge_tensor = preprocessed_cotopaxi['tensors']['edge']
edge_stats = get_edge_stats(edge_tensor)
print_edge_stats(edge_stats)
visualize_edge_sequence(
    edge_tensor,
    timestamps,
    numeric_labels,
    example_indices,  # Using the same example_indices we created before
    label_mapping
)

# %%
# Delete your tensor variables
del corrected_tensor
del edge_tensor
del preprocessed_cotopaxi

# Run garbage collection
gc.collect()

# If using CUDA, clear the CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ## Reventador Tensors

# %% [markdown]
# ### Data Loader

# %%
base_directories_REVENTADOR = [
    os.path.join(project_root, 'data/input/Reventador/Nublado'),
    os.path.join(project_root, 'data/input/Reventador/Emisiones'),
    os.path.join(project_root, 'data/input/Reventador/Flujo')
]

# Initialize data loader
print("\nInitializing data loader...")
thermal_loader_reventador = ThermalDataLoader(
    base_directories=base_directories_REVENTADOR,
    limit_per_directory=4500,
    allowed_labels=allowed_labels
)

# %% [markdown]
# ### Create dataset

# %%
# Create initial dataset
print("\nCreating Revemtador dataset...")
dataset_reventador = thermal_loader_reventador.create_dataset()

# %%
reventador_thresholds = {'low': 1.00, 'medium': 10.00, 'high': 30.0, 'very_high': 60.00}
preprocessed_reventador = preprocess_dataset(dataset_reventador, thresholds=reventador_thresholds)

# %%
save_preprocessed_dataset(preprocessed_reventador, filename='reventador_preprocessed.pt')

# %% [markdown]
# ### Data Visualization

# %% [markdown]
# #### Thermal Sequence

# %%
# Extract data for visualization
viz_data_reventador = extract_visualization_data(preprocessed_reventador)

# Use the extracted data (replaces your original lines)
timestamps = viz_data_reventador['timestamps']
numeric_labels = viz_data_reventador['numeric_labels']
label_mapping = viz_data_reventador['label_mapping']
corrected_tensor = viz_data_reventador['corrected_tensor']

label_examples = get_label_examples(numeric_labels, label_mapping)
example_indices = list(label_examples.values())

thermal_stats = get_thermal_stats(corrected_tensor)
print_thermal_stats(thermal_stats)
visualize_thermal_sequence(
    corrected_tensor,
    timestamps,
    numeric_labels,
    example_indices,
    thermal_stats['global_min'],
    thermal_stats['global_max'],
    label_mapping
)

# %% [markdown]
# #### Thresholds Sequence

# %%
visualize_thermal_threshold_comparison(
    corrected_tensor,
    timestamps,
    numeric_labels,
    example_indices,
    label_mapping,
    thresholds=reventador_thresholds
)

# %% [markdown]
# #### Edges Sequence

# %%
# Edge detection visualization and stats
edge_tensor = preprocessed_reventador['tensors']['edge']
edge_stats = get_edge_stats(edge_tensor)
print_edge_stats(edge_stats)
visualize_edge_sequence(
    edge_tensor,
    timestamps,
    numeric_labels,
    example_indices,  # Using the same example_indices we created before
    label_mapping
)

# %%
# Delete your tensor variables
del corrected_tensor
del edge_tensor
del preprocessed_reventador

# Run garbage collection
gc.collect()

# If using CUDA, clear the CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %%



