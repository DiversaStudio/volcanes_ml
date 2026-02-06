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

def visualize_thermal_sequence(tensor, timestamps, labels, indices, global_min, global_max, label_mapping,
                               use_percentile=True, percentile_range=(2, 98), cmap='inferno'):
    """
    Visualize a sequence of thermal images with improved contrast.

    :param tensor: Thermal image tensor
    :param timestamps: List of timestamps
    :param labels: Numeric labels
    :param indices: Indices of images to visualize
    :param global_min: Global minimum temperature (used if use_percentile=False)
    :param global_max: Global maximum temperature (used if use_percentile=False)
    :param label_mapping: Dictionary mapping label names to numeric indices
    :param use_percentile: If True, uses percentile-based normalization for better contrast
    :param percentile_range: Tuple of (low_percentile, high_percentile) for normalization
    :param cmap: Colormap to use (default: 'inferno')
    """
    inv_label_mapping = {v: k for k, v in label_mapping.items()}

    n_images = len(indices)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))

    # Handle single image case
    if n_images == 1:
        axes = [axes]

    # Determine vmin and vmax for better contrast
    if use_percentile:
        tensor_np = tensor.cpu().numpy()
        vmin = np.percentile(tensor_np, percentile_range[0])
        vmax = np.percentile(tensor_np, percentile_range[1])
    else:
        vmin = global_min
        vmax = global_max

    for i, idx in enumerate(indices):
        im = axes[i].imshow(tensor[:, :, idx].cpu(), cmap=cmap,
                            vmin=vmin, vmax=vmax)

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

def visualize_thermal_threshold_comparison(thermal_tensor, timestamps, labels, indices, label_mapping, thresholds=None):
    """Visualize thermal images and their thresholded versions."""
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    
    n_images = len(indices)
    fig, axes = plt.subplots(5, n_images, figsize=(5 * n_images, 25))
    
    vmin, vmax = thermal_tensor.min().item(), thermal_tensor.max().item()
    threshold_data = create_thermal_threshold_tensor(thermal_tensor, thresholds=thresholds)
    
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

# Academic paper visualization
def visualize_feature_composition(
    thermal_tensor,
    edge_tensor,
    threshold_tensors,
    timestamps,
    labels,
    indices,
    label_mapping,
    threshold_name='high',
    threshold_value=None,
    figsize_per_col=4,
    dpi=150,
    cmap_thermal='inferno',
    show_colorbar=True,
    title_fontsize=11,
    label_fontsize=10,
    save_path=None,
    use_percentile=True,
    percentile_range=(2, 98)
):
    """
    Create a professional 3-row composition for academic papers showing:
    - Row 1: Original thermal images
    - Row 2: Canny edge detection
    - Row 3: Selected thermal threshold

    :param thermal_tensor: Thermal image tensor (H, W, T)
    :param edge_tensor: Edge detection tensor (H, W, T)
    :param threshold_tensors: Dictionary of threshold tensors {'name': tensor}
    :param timestamps: List of timestamps
    :param labels: Numeric labels tensor
    :param indices: List of image indices to display
    :param label_mapping: Dictionary mapping label names to numeric indices
    :param threshold_name: Name of threshold to display ('low', 'medium', 'high', 'very_high')
    :param threshold_value: Temperature value for the threshold (for label display)
    :param figsize_per_col: Width per column in inches
    :param dpi: Figure resolution
    :param cmap_thermal: Colormap for thermal images
    :param show_colorbar: Whether to show colorbars
    :param title_fontsize: Font size for titles
    :param label_fontsize: Font size for labels
    :param save_path: Path to save the figure (optional)
    :param use_percentile: If True, uses percentile-based normalization for better contrast
    :param percentile_range: Tuple of (low_percentile, high_percentile) for normalization
    :return: Figure and axes objects
    """
    inv_label_mapping = {v: k for k, v in label_mapping.items()}

    n_images = len(indices)
    fig_width = figsize_per_col * n_images + (1.5 if show_colorbar else 0)
    fig_height = figsize_per_col * 2.2  # Reduced height for tighter rows

    fig, axes = plt.subplots(3, n_images, figsize=(fig_width, fig_height), dpi=dpi,
                             gridspec_kw={'hspace': 0.12, 'wspace': 0.05})

    # Ensure axes is 2D even for single image
    if n_images == 1:
        axes = axes.reshape(3, 1)

    # Get thermal range for consistent coloring
    if use_percentile:
        tensor_np = thermal_tensor.cpu().numpy()
        vmin = np.percentile(tensor_np, percentile_range[0])
        vmax = np.percentile(tensor_np, percentile_range[1])
    else:
        vmin = thermal_tensor.min().item()
        vmax = thermal_tensor.max().item()

    # Get selected threshold tensor
    if threshold_name not in threshold_tensors:
        available = list(threshold_tensors.keys())
        raise ValueError(f"Threshold '{threshold_name}' not found. Available: {available}")
    threshold_tensor = threshold_tensors[threshold_name]

    # Row labels
    row_labels = [
        'Thermal Image',
        'Canny Edge Detection',
        f'Threshold: T > {threshold_value}°C' if threshold_value else f'Threshold: {threshold_name}'
    ]

    for i, idx in enumerate(indices):
        # Get label text
        if isinstance(labels, torch.Tensor):
            label_val = labels[idx].item()
        else:
            label_val = labels[idx]
        label_text = inv_label_mapping.get(label_val, str(label_val))

        # Column letter identifier
        col_letter = chr(ord('a') + i)

        # Row 1: Thermal image
        thermal_img = thermal_tensor[:, :, idx].cpu().numpy()
        im1 = axes[0, i].imshow(thermal_img, cmap=cmap_thermal, vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'{timestamps[idx]}\n{label_text}', fontsize=title_fontsize)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].text(0.95, 0.95, f'{col_letter}1', transform=axes[0, i].transAxes,
                       fontsize=label_fontsize+2, fontweight='bold', va='top', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))

        if i == 0:
            axes[0, i].set_ylabel(row_labels[0], fontsize=label_fontsize)

        # Row 2: Edge detection (inverted: white edges on black background)
        edge_img = edge_tensor[:, :, idx].cpu().numpy()
        im2 = axes[1, i].imshow(edge_img, cmap='gray_r', vmin=0, vmax=255)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[1, i].text(0.95, 0.95, f'{col_letter}2', transform=axes[1, i].transAxes,
                       fontsize=label_fontsize+2, fontweight='bold', va='top', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))

        if i == 0:
            axes[1, i].set_ylabel(row_labels[1], fontsize=label_fontsize)

        # Row 3: Threshold (inverted: white for hot regions)
        threshold_img = threshold_tensor[:, :, idx].cpu().numpy()
        im3 = axes[2, i].imshow(threshold_img, cmap='gray_r', vmin=0, vmax=255)
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])
        axes[2, i].text(0.95, 0.95, f'{col_letter}3', transform=axes[2, i].transAxes,
                       fontsize=label_fontsize+2, fontweight='bold', va='top', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))

        if i == 0:
            axes[2, i].set_ylabel(row_labels[2], fontsize=label_fontsize)

    # Apply tight layout first to get proper positioning
    plt.tight_layout(rect=[0, 0, 0.9 if show_colorbar else 1, 1], h_pad=0.5)

    # Add colorbar aligned with thermal row
    if show_colorbar:
        # Get the position of the first row axes to align colorbar
        pos0 = axes[0, -1].get_position()
        cbar_ax = fig.add_axes([0.92, pos0.y0, 0.015, pos0.height])
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.set_label('Temperature (°C)', fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=9)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig, axes