import torch
from torch.utils.data import Dataset
import os

class ThermalDataset(Dataset):
    """Custom Dataset for thermal images with multiple feature tensors."""
    
    def __init__(self, preprocessed_data, indices=None):
        """Initialize dataset with processed tensors."""
        # Store indices for data splitting
        self.indices = indices if indices is not None else range(len(preprocessed_data['tensors']['corrected']))
        
        # Store the original data structure
        self.data = preprocessed_data
        
        if indices is not None:
            # For split datasets, only keep selected indices
            self.corrected_tensor = preprocessed_data['tensors']['corrected'][indices]
            self.edge_tensor = preprocessed_data['tensors']['edge'][indices]
            self.edge_features = preprocessed_data['tensors']['edge_features'][indices] if 'edge_features' in preprocessed_data['tensors'] else None
            self.threshold_tensors = {
                level: preprocessed_data['tensors']['threshold'][level][indices] 
                for level in preprocessed_data['tensors']['threshold'].keys()
            }
            self.labels = preprocessed_data['labels']['numeric_labels'][indices]
        else:
            # For full dataset
            self.corrected_tensor = preprocessed_data['tensors']['corrected']
            self.edge_tensor = preprocessed_data['tensors']['edge']
            self.edge_features = preprocessed_data['tensors']['edge_features'] if 'edge_features' in preprocessed_data['tensors'] else None
            self.threshold_tensors = preprocessed_data['tensors']['threshold']
            self.labels = preprocessed_data['labels']['numeric_labels']
        
        # Store metadata and mapping
        self.metadata = preprocessed_data['metadata']
        self.label_mapping = preprocessed_data['labels']['label_mapping']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get base tensors
        corrected = self.corrected_tensor[idx]
        edge = self.edge_tensor[idx]
        
        # Ensure correct shape for CNNs [C, H, W]
        if len(corrected.shape) == 2:  # If [H, W]
            corrected = corrected.unsqueeze(0)  # Add channel dimension
            edge = edge.unsqueeze(0)
        
        # Handle thresholds
        thresholds = {}
        for name, tensor in self.threshold_tensors.items():
            thresh = tensor[idx]
            if len(thresh.shape) == 2:  # If [H, W]
                thresh = thresh.unsqueeze(0)  # Add channel dimension
            thresholds[name] = thresh
        
        sample = {
            'corrected': corrected,
            'edge': edge,
            'thresholds': thresholds,
            'label': self.labels[idx]
        }
        
        # Add edge_features if they exist
        if self.edge_features is not None:
            sample['edge_features'] = self.edge_features[idx]
            
        return sample
    
# Save datasets with memory optimization
def save_large_dataset(dataset, path, name="dataset"):
    """Save large dataset by saving components separately."""
    print(f"\nSaving {name}...")

    # Create directory for this dataset
    dataset_dir = os.path.join(path, name)
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        # Save tensors separately
        tensor_dir = os.path.join(dataset_dir, 'tensors')
        os.makedirs(tensor_dir, exist_ok=True)

        # Save corrected tensor
        print(f"Saving corrected tensor with {dataset.corrected_tensor.size(0)} samples...")
        torch.save(dataset.corrected_tensor, os.path.join(tensor_dir, 'corrected.pt'))

        # Save edge tensor
        print(f"Saving edge tensor with {dataset.edge_tensor.size(0)} samples...")
        torch.save(dataset.edge_tensor, os.path.join(tensor_dir, 'edge.pt'))

        # Save threshold tensors
        print("Saving threshold tensors...")
        threshold_dir = os.path.join(tensor_dir, 'threshold')
        os.makedirs(threshold_dir, exist_ok=True)
        for level, tensor in dataset.threshold_tensors.items():
            print(f"Saving threshold tensor '{level}' with {tensor.size(0)} samples...")
            torch.save(tensor, os.path.join(threshold_dir, f'{level}.pt'))

        # Save labels and metadata
        print(f"Saving labels and metadata with {len(dataset.labels)} samples...")
        torch.save({
            'labels': dataset.labels,
            'indices': dataset.indices,
        }, os.path.join(dataset_dir, 'metadata.pt'))

        print(f"{name} saved successfully!")
        return True

    except Exception as e:
        print(f"Error saving {name}: {str(e)}")
        return False
    
def load_large_dataset(path, name="dataset"):
    """Load a large dataset by loading components separately."""
    print(f"\nLoading {name}...")

    # Define the paths
    tensor_dir = os.path.join(path, 'tensors')
    threshold_dir = os.path.join(tensor_dir, 'threshold')

    try:
        # Load the main tensors
        print("Loading corrected tensor...")
        corrected_tensor = torch.load(os.path.join(tensor_dir, 'corrected.pt'))
        print(f"Corrected tensor loaded with {corrected_tensor.size(0)} samples.")

        print("Loading edge tensor...")
        edge_tensor = torch.load(os.path.join(tensor_dir, 'edge.pt'))
        print(f"Edge tensor loaded with {edge_tensor.size(0)} samples.")

        # Load the threshold tensors
        print("Loading threshold tensors...")
        threshold_tensors = {}
        for filename in os.listdir(threshold_dir):
            level = filename.replace('.pt', '')
            threshold_tensors[level] = torch.load(os.path.join(threshold_dir, filename))
            print(f"Threshold tensor '{level}' loaded with {threshold_tensors[level].size(0)} samples.")

        # Load labels and metadata
        print("Loading labels and metadata...")
        metadata = torch.load(os.path.join(path, 'metadata.pt'))
        labels = metadata['labels']
        indices = metadata['indices']
        print(f"Labels loaded with {len(labels)} samples.")

        # Check consistency
        num_samples = len(labels)
        assert corrected_tensor.size(0) == num_samples, "Mismatch in corrected tensor size"
        assert edge_tensor.size(0) == num_samples, "Mismatch in edge tensor size"
        for level, tensor in threshold_tensors.items():
            assert tensor.size(0) == num_samples, f"Mismatch in threshold tensor '{level}' size"

        # Reconstruct the dataset
        dataset = {
            'corrected_tensor': corrected_tensor,
            'edge_tensor': edge_tensor,
            'threshold_tensors': threshold_tensors,
            'labels': labels,
            'indices': indices
        }

        print(f"{name} loaded successfully!")
        return dataset

    except Exception as e:
        print(f"Error loading {name}: {str(e)}")
        return None


class CustomTensorDataset(Dataset):
    def __init__(self, dataset):
        self.corrected_tensor = dataset['corrected_tensor']
        self.edge_tensor = dataset['edge_tensor']
        self.threshold_tensors = dataset['threshold_tensors']
        self.labels = dataset['labels']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            # Add a channel dimension for corrected and edge tensors
            'corrected': self.corrected_tensor[idx].unsqueeze(0),  # Shape: [1, 480, 640]
            'edge': self.edge_tensor[idx].unsqueeze(0),            # Shape: [1, 480, 640]
            
            # Concatenate threshold tensors along the channel dimension
            'thresholds': torch.cat([
                self.threshold_tensors['low'][idx].unsqueeze(0),
                self.threshold_tensors['medium'][idx].unsqueeze(0),
                self.threshold_tensors['high'][idx].unsqueeze(0),
                self.threshold_tensors['very_high'][idx].unsqueeze(0)
            ], dim=0),  # Shape: [4, 480, 640]
            
            'label': self.labels[idx]
        }
        return sample

