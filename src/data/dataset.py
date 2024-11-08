import torch
from torch.utils.data import Dataset

class ThermalDataset(Dataset):
    """Custom Dataset for thermal images with multiple feature tensors."""
    
    def __init__(self, dataset_dict):
        """
        Initialize dataset with processed tensors.
        """
        # Handle either numpy array or torch tensor for corrected tensor
        if torch.is_tensor(dataset_dict['tensors']['corrected']):
            self.corrected_tensor = dataset_dict['tensors']['corrected']
        else:
            self.corrected_tensor = torch.from_numpy(dataset_dict['tensors']['corrected'])
        self.corrected_tensor = self.corrected_tensor.permute(2, 0, 1).unsqueeze(1).float()
            
        # Handle either numpy array or torch tensor for edge tensor
        if torch.is_tensor(dataset_dict['tensors']['edge']):
            self.edge_tensor = dataset_dict['tensors']['edge']
        else:
            self.edge_tensor = torch.from_numpy(dataset_dict['tensors']['edge'])
        self.edge_tensor = self.edge_tensor.permute(2, 0, 1).unsqueeze(1).float()
            
        # Handle either numpy array or torch tensor for temporal tensor
        if torch.is_tensor(dataset_dict['tensors']['temporal']):
            self.temporal_tensor = dataset_dict['tensors']['temporal']
        else:
            self.temporal_tensor = torch.from_numpy(dataset_dict['tensors']['temporal'])
        self.temporal_tensor = self.temporal_tensor.permute(2, 3, 0, 1).float()
            
        # Handle either numpy array or torch tensor for threshold tensor
        if torch.is_tensor(dataset_dict['tensors']['threshold']['low']):
            self.threshold_tensor = dataset_dict['tensors']['threshold']['low']
        else:
            self.threshold_tensor = torch.from_numpy(dataset_dict['tensors']['threshold']['low'])
        self.threshold_tensor = self.threshold_tensor.permute(2, 0, 1).unsqueeze(1).float()
        
        # Handle labels
        if 'labels' in dataset_dict:
            if torch.is_tensor(dataset_dict['labels']['numeric_labels']):
                self.labels = dataset_dict['labels']['numeric_labels']
            else:
                self.labels = torch.tensor(dataset_dict['labels']['numeric_labels'])
        
        # Store metadata but don't return in __getitem__
        self.timestamps = dataset_dict['metadata']['timestamps']
        self.filenames = dataset_dict['metadata']['filenames']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Don't return timestamp and filename in the batch
        return {
            'corrected': self.corrected_tensor[idx],
            'edge': self.edge_tensor[idx],
            'temporal': self.temporal_tensor[idx],
            'threshold': self.threshold_tensor[idx],
            'label': self.labels[idx]
        }

    # Add methods to access metadata if needed
    def get_timestamp(self, idx):
        return self.timestamps[idx]

    def get_filename(self, idx):
        return self.filenames[idx]