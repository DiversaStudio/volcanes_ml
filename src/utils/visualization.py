import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.features.thermal import create_thermal_threshold_tensor

def get_label_examples(labels, label_mapping):
    """
    Get example indices for each label class.
    
    :param labels: Tensor of numeric labels
    :param label_mapping: Dictionary mapping label names to numeric indices
    :return: Dictionary mapping label indices to example indices
    """
    label_examples = {}
    for label in label_mapping.values():
        label_indices = [i for i, l in enumerate(labels) if l.item() == label]
        if label_indices:
            label_examples[label] = label_indices[0]
    return label_examples

# Thermal visualization functions
def get_thermal_stats(tensor):
    """
    Calculate basic thermal statistics from tensor.
    
    :param tensor: PyTorch tensor of thermal images
    :return: Dictionary with statistics
    """
    return {
        'global_min': torch.min(tensor).item(),
        'global_max': torch.max(tensor).item(),
        'mean_temp': torch.mean(tensor).item(),
        'std_temp': torch.std(tensor).item()
    }

def print_thermal_stats(stats):
    """Print thermal statistics in a formatted way."""
    print("Temperature Statistics:")
    print(f"Global min: {stats['global_min']:.2f}°C")
    print(f"Global max: {stats['global_max']:.2f}°C")
    print(f"Mean temperature: {stats['mean_temp']:.2f}°C")
    print(f"Standard deviation: {stats['std_temp']:.2f}°C")

def visualize_thermal_sequence(tensor, timestamps, labels, indices, global_min, global_max, label_mapping):
    """Visualize a sequence of thermal images."""
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    
    n_images = len(indices)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    for i, idx in enumerate(indices):
        im = axes[i].imshow(tensor[:, :, idx].cpu(), cmap='inferno', 
                            vmin=global_min, vmax=global_max)
        
        label_text = inv_label_mapping[labels[idx].item()] if isinstance(labels, torch.Tensor) else inv_label_mapping[labels[idx]]
        axes[i].set_title(f'Time: {timestamps[idx]}\nLabel: {label_text}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    plt.suptitle('Thermal Image Sequence', y=1.05)
    plt.tight_layout()
    plt.show()

# Edge detection visualization functions
def get_edge_stats(edge_features):
    """
    Calculate edge detection statistics.
    
    :param edge_features: Edge features tensor
    :return: Dictionary with statistics
    """
    return {
        'edge_density': torch.mean(edge_features[:, 0]).item(),
        'total_edges': torch.mean(edge_features[:, 1]).item(),
        'non_zero_ratio': torch.mean(edge_features[:, 2]).item()
    }

def print_edge_stats(stats):
    """Print edge detection statistics."""
    print("\nEdge Detection Statistics:")
    print(f"Average edge density: {stats['edge_density']:.4f}")
    print(f"Average total edges: {stats['total_edges']:.2f}")
    print(f"Average non-zero ratio: {stats['non_zero_ratio']:.4f}")

def visualize_edge_sequence(tensor, timestamps, labels, indices, label_mapping):
    """Visualize a sequence of edge detection images."""
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    
    n_images = len(indices)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    for i, idx in enumerate(indices):
        edge_image = tensor[:, :, idx].cpu().numpy()
        
        im = axes[i].imshow(edge_image, cmap='binary', vmin=0, vmax=255)
        
        label_text = inv_label_mapping[labels[idx].item()] if isinstance(labels, torch.Tensor) else inv_label_mapping[labels[idx]]
        axes[i].set_title(f'Time: {timestamps[idx]}\nLabel: {label_text}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    plt.suptitle('Edge Detection Sequence', y=1.05)
    plt.tight_layout()
    plt.show()

# Threshold visualization functions
def visualize_thermal_threshold_comparison(thermal_tensor, timestamps, labels, indices, label_mapping):
    """Visualize thermal images and their thresholded versions."""
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    
    n_images = len(indices)
    fig, axes = plt.subplots(5, n_images, figsize=(5 * n_images, 25))
    
    vmin, vmax = thermal_tensor.min().item(), thermal_tensor.max().item()
    threshold_data = create_thermal_threshold_tensor(thermal_tensor)
    
    for i, idx in enumerate(indices):
        thermal_image = thermal_tensor[:, :, idx].cpu().numpy()
        
        # Original thermal image
        im1 = axes[0, i].imshow(thermal_image, cmap='inferno', vmin=vmin, vmax=vmax)
        label_text = inv_label_mapping[labels[idx].item()] if isinstance(labels, torch.Tensor) else inv_label_mapping[labels[idx]]
        axes[0, i].set_title(f'Time: {timestamps[idx]}\nLabel: {label_text}')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        divider = make_axes_locatable(axes[0, i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax, label='Temperature (°C)')
        
        # Threshold images
        for j, (name, threshold) in enumerate(threshold_data['thresholds'].items(), 1):
            threshold_image = threshold_data['tensors'][name][:, :, idx].cpu().numpy()
            axes[j, i].imshow(threshold_image, cmap='binary', vmin=0, vmax=255)
            axes[j, i].set_title(f'Threshold: T > {threshold}°C')
            axes[j, i].set_xticks([])
            axes[j, i].set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
# Plot class distribution
def plot_class_distribution(train_dataset, val_dataset):
    plt.figure(figsize=(10, 5))
    
    # Get distributions
    train_labels = [data['label'].item() for data in train_dataset]
    val_labels = [data['label'].item() for data in val_dataset]
    
    unique_labels = sorted(set(train_labels))
    train_counts = [train_labels.count(l) for l in unique_labels]
    val_counts = [val_labels.count(l) for l in unique_labels]
    
    x = np.arange(len(unique_labels))
    width = 0.35
    
    plt.bar(x - width/2, train_counts, width, label='Train')
    plt.bar(x + width/2, val_counts, width, label='Validation')
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Train and Validation Sets')
    plt.xticks(x, unique_labels)
    plt.legend()
    plt.tight_layout()
    plt.show()