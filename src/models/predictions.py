import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from flirpy.io.fff import Fff

from src.models.multibranch import MultiBranchModel
from src.features.edge_detection import process_canny_image
from src.features.thermal import create_thermal_threshold_tensor

def load_model(model_path, device):
    """Load trained model from checkpoint."""
    model = MultiBranchModel(n_classes=4).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, {
        'Despejado': 0,  # Clear/No activity
        'Nublado': 1,    # Cloudy conditions
        'Emisiones': 2,  # Emissions
        'Flujo': 3       # Lava flow
    }

def prepare_single_image(thermal_image, device):
    """Prepare a single thermal image for inference."""
    thermal_tensor = torch.tensor(thermal_image).float().unsqueeze(0).unsqueeze(0)
    
    edge_image, _ = process_canny_image(thermal_image)
    edge_tensor = torch.tensor(edge_image).float().unsqueeze(0).unsqueeze(0)
    
    threshold_data = create_thermal_threshold_tensor(thermal_tensor.squeeze())
    threshold_tensor = torch.cat([
        threshold_data['tensors'][level] for level in ['low', 'medium', 'high', 'very_high']
    ], dim=0).unsqueeze(0)
    
    return {
        'corrected': thermal_tensor.to(device),
        'edge': edge_tensor.to(device),
        'thresholds': threshold_tensor.to(device)
    }, edge_image, threshold_data

def predict_image(model, processed_image, label_mapping):
    """Get prediction and confidence for a single image."""
    with torch.no_grad():
        outputs = model(
            processed_image['corrected'],
            processed_image['edge'],
            processed_image['thresholds']
        )
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
    
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    predicted_label = inv_label_mapping[prediction.item()]
    
    return {
        'label': predicted_label,
        'confidence': confidence.item() * 100,
        'probabilities': {
            inv_label_mapping[i]: prob.item() * 100 
            for i, prob in enumerate(probabilities[0])
        }
    }

def visualize_prediction(thermal_image, edge_image, threshold_data, metadata, prediction_results):
    """Visualize thermal image with prediction results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    im1 = axes[0,0].imshow(thermal_image, cmap='inferno')
    axes[0,0].set_title('Thermal Image')
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax, label='Temperature (°C)')
    
    axes[0,1].imshow(edge_image, cmap='binary')
    axes[0,1].set_title('Edge Detection')
    
    threshold_img = threshold_data['tensors']['high'].squeeze().numpy()
    axes[1,0].imshow(threshold_img, cmap='binary')
    axes[1,0].set_title('High Temperature Threshold')
    
    axes[1,1].axis('off')
    result_text = (
        f"Prediction Results\n\n"
        f"Label: {prediction_results['label']}\n"
        f"Confidence: {prediction_results['confidence']:.2f}%\n\n"
        f"Probabilities:\n"
    )
    for label, prob in prediction_results['probabilities'].items():
        result_text += f"{label}: {prob:.2f}%\n"
    
    if metadata:
        result_text += f"\nMetadata:\n"
        result_text += f"Filename: {metadata.get('file_path', 'N/A')}\n"
        result_text += f"Date: {metadata.get('Datetime (UTC)', 'N/A')}"
    
    axes[1,1].text(0.05, 0.95, result_text, transform=axes[1,1].transAxes,
                  verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def process_single_file(model, file_path, label_mapping, device='cpu', output_dir=None):
    """Process a single FFF file and return/save predictions."""
    try:
        fff_reader = Fff(file_path)
        thermal_image = fff_reader.get_image()
        metadata = fff_reader.meta
        metadata['file_path'] = os.path.basename(file_path)
        
        processed_image, edge_image, threshold_data = prepare_single_image(thermal_image, device)
        prediction = predict_image(model, processed_image, label_mapping)
        
        if output_dir:
            fig = visualize_prediction(thermal_image, edge_image, threshold_data, metadata, prediction)
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            fig.savefig(os.path.join(output_dir, f"{filename}_prediction.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return {
            'filename': metadata['file_path'],
            'datetime': metadata.get('Datetime (UTC)', 'N/A'),
            'predicted_label': prediction['label'],
            'confidence': prediction['confidence'],
            **{f"prob_{k}": v for k, v in prediction['probabilities'].items()}
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_directory(model, directory_path, label_mapping, device='cpu', output_dir=None):
    """Process all FFF files in a directory and save predictions."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    results = []
    fff_files = [f for f in os.listdir(directory_path) if f.endswith('.fff')]
    
    for filename in tqdm(fff_files, desc="Processing images"):
        file_path = os.path.join(directory_path, filename)
        result = process_single_file(model, file_path, label_mapping, device, 
                                   os.path.join(output_dir, 'plots') if output_dir else None)
        if result:
            results.append(result)
    
    results_df = pd.DataFrame(results)
    if output_dir:
        results_df.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)
    
    return results_df