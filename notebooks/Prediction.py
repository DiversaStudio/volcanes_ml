# %%
# Import libraries
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# %%
# Set path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# %%
# Import necessary modules
from src.data.loader import ThermalDataLoader
from src.features.thermal import create_thermal_threshold_tensor
from src.features.edge_detection import create_edge_detection_tensors
from src.models.multibranch import MultiBranchModel
from src.data.dataset import ThermalDataset
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

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
def preprocess_data(data_path, is_single_file=False):
    """
    Preprocess new thermal data for prediction following the original pipeline.
    """
    # Initialize data loader
    if is_single_file:
        base_directories = [os.path.dirname(data_path)]
    else:
        base_directories = [data_path]
        
    thermal_loader = ThermalDataLoader(
        base_directories=base_directories,
        allowed_labels=None  # No labels needed for prediction
    )
    
    # Create initial dataset
    print("\nCreating dataset...")
    dataset = thermal_loader.create_dataset()
    
    if dataset is not None:
        # Follow the same preprocessing steps as in DataPreprocessing.py
        print("\nPreprocessing data...")
        
        # Convert to tensor and handle NaN values
        corrected_tensor = torch.tensor(dataset['tensors']['corrected']).float()
        if torch.isnan(corrected_tensor).any():
            print("Warning: NaN values found in corrected tensor!")
            corrected_tensor = torch.nan_to_num(corrected_tensor, nan=0.0)
        
        # Create edge detection tensors
        print("Creating edge detection tensors...")
        edge_detection_data = create_edge_detection_tensors(corrected_tensor)
        edge_tensor = torch.tensor(edge_detection_data['edge_tensor']).float()
        edge_tensor = torch.nan_to_num(edge_tensor, nan=0.0)
        
        # Create threshold tensors
        print("Creating threshold tensors...")
        threshold_data = create_thermal_threshold_tensor(corrected_tensor)
        for name, tensor in threshold_data['tensors'].items():
            threshold_data['tensors'][name] = torch.nan_to_num(tensor, nan=0.0)
        
        # Create preprocessed dataset dictionary (following original structure)
        preprocessed_dataset = {
            'tensors': {
                'corrected': corrected_tensor,
                'edge': edge_tensor,
                'threshold': threshold_data['tensors']
            },
            'metadata': dataset['metadata']
        }
        
        # Apply the same permutations as in FeatureEngineering.py
        print("\nReorganizing tensors...")
        preprocessed_dataset['tensors']['corrected'] = preprocessed_dataset['tensors']['corrected'].permute(2, 0, 1)
        preprocessed_dataset['tensors']['edge'] = preprocessed_dataset['tensors']['edge'].permute(2, 0, 1)
        
        # Handle threshold tensors permutation
        threshold_tensors = []
        for level in preprocessed_dataset['tensors']['threshold'].keys():
            permuted_tensor = preprocessed_dataset['tensors']['threshold'][level].permute(2, 0, 1)
            threshold_tensors.append(permuted_tensor)
        
        # Stack threshold tensors
        stacked_thresholds = torch.stack(threshold_tensors, dim=1)
        
        # Add channel dimension for corrected and edge tensors
        corrected = preprocessed_dataset['tensors']['corrected'].unsqueeze(1)
        edge = preprocessed_dataset['tensors']['edge'].unsqueeze(1)
        
        print("\nFinal tensor shapes:")
        print(f"Corrected: {corrected.shape}")
        print(f"Edge: {edge.shape}")
        print(f"Thresholds: {stacked_thresholds.shape}")
        
        return {
            'corrected': corrected,
            'edge': edge,
            'thresholds': stacked_thresholds,
            'metadata': dataset['metadata']
        }
    
    return None

# %%
def load_model_and_predict(model_path, preprocessed_data):
    """
    Load the trained model and make predictions.
    """
    print("\nLoading model and making predictions...")
    
    # Load model
    n_classes = 3
    model = MultiBranchModel(n_classes).to(device)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare data
    with torch.no_grad():
        corrected = preprocessed_data['corrected'].float().to(device)
        edge = preprocessed_data['edge'].float().to(device)
        thresholds = preprocessed_data['thresholds'].float().to(device)
        
        # Get predictions
        outputs = model(corrected, edge, thresholds)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.cpu().numpy(), probabilities.cpu().numpy()


# %%
# Define label mapping
label_mapping = {
    0: 'Despejado',  # Clear
    1: 'Nublado',    # Cloudy
    2: 'Emisiones'   # Emissions
}

# %%
if __name__ == "__main__":
    # Set paths
    model_path = os.path.join(project_root, 'data', 'processed', 'best_model.pt')
    data_path = os.path.join(project_root, 'data', 'test', 'DiarioRaw')  # Replace with your test data path
    
    # Process data
    preprocessed_data = preprocess_data(data_path, is_single_file=False)
    
    if preprocessed_data is not None:
        # Make predictions
        predictions, probabilities = load_model_and_predict(model_path, preprocessed_data)
        
        # Print results
        print("\nPrediction Summary:")
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        for pred, count in zip(unique_predictions, counts):
            print(f"{label_mapping[pred]}: {count} images ({count/len(predictions)*100:.2f}%)")
        
        # Save results
        results = {
            'predictions': predictions,
            'probabilities': probabilities
        }
        results_path = os.path.join(project_root, 'data', 'results', 'predictions.pt')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        torch.save(results, results_path)
        print(f"\nResults saved to: {results_path}")

# %%
def export_predictions_to_csv(predictions, probabilities, metadata, input_dir, save_dir):
    """
    Export predictions to CSV files.
    
    Args:
        predictions: numpy array of predicted classes
        probabilities: numpy array of prediction probabilities
        metadata: dictionary containing timestamps and file information
        save_dir: directory to save CSV files
    """
        
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Current timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Get file paths and timestamps from metadata
    file_timestamps = metadata['timestamps']  # This should be a list of timestamps from the files
    # Get list of files from input directory
    filenames = sorted([f for f in os.listdir(input_dir) if f.endswith('.fff')])
    
    # 1. Detailed predictions
    detailed_results = []
    for fname, pred, probs in zip(filenames, predictions, probabilities):
        detailed_results.append({
            'file_name': fname,
            'prediction': label_mapping[pred],
            'confidence': probs[pred] * 100,  # Convert to percentage
            'prob_despejado': probs[0] * 100,
            'prob_nublado': probs[1] * 100,
            'prob_emisiones': probs[2] * 100
        })
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = os.path.join(save_dir, f'detailed_predictions_{timestamp}.csv')
    detailed_df.to_csv(detailed_path, index=False)
    
    # 2. Summary statistics
    unique_predictions, counts = np.unique(predictions, return_counts=True)
    summary_results = []
    for pred, count in zip(unique_predictions, counts):
        summary_results.append({
            'class': label_mapping[pred],
            'count': int(count),
            'percentage': float(count/len(predictions)*100)
        })
    
    # Save summary
    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(save_dir, f'summary_predictions_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nDetailed predictions saved to: {detailed_path}")
    print(f"Summary saved to: {summary_path}")
    
    # Also print summary to console
    print("\nPrediction Summary:")
    print(summary_df.to_string(index=False))

# %%
if preprocessed_data is not None:
    # Make predictions
    predictions, probabilities = load_model_and_predict(model_path, preprocessed_data)
    
    # Export to CSV
    save_dir = os.path.join(project_root, 'data', 'results')
    export_predictions_to_csv(predictions, probabilities, preprocessed_data['metadata'], os.path.join(project_root, 'data', 'test', 'DiarioRaw'), save_dir)


