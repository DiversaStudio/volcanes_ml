import cv2
import numpy as np
import torch

def apply_canny(image, low_threshold=100, high_threshold=200):
    """Apply Canny edge detection to an image."""
    return cv2.Canny(image.astype(np.uint8), low_threshold, high_threshold)

def calculate_edge_statistics(edge_image):
    """Calculate statistical measures for an edge-detected image."""
    return {
        'edge_density': np.mean(edge_image) / 255.0,
        'total_edges': np.sum(edge_image) / 255,
        'non_zero_ratio': np.count_nonzero(edge_image) / edge_image.size
    }

def process_canny_image(thermal_image):
    """Process a thermal image with Canny edge detection and calculate statistics."""
    if isinstance(thermal_image, torch.Tensor):
        thermal_image = thermal_image.cpu().numpy()

    normalized_image = cv2.normalize(thermal_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    equalized_image = cv2.equalizeHist(normalized_image)
    canny_image = apply_canny(equalized_image)
    
    global_stats = calculate_edge_statistics(canny_image)
    
    return canny_image, global_stats

def create_edge_detection_tensors(thermal_tensor):
    """Create edge detection tensor and statistical features from thermal tensor."""
    height, width, num_images = thermal_tensor.shape
    
    edge_tensor = np.zeros((height, width, num_images), dtype=np.uint8)
    edge_features = np.zeros((num_images, 3))
    
    for i in range(num_images):
        edge_image, global_stats = process_canny_image(thermal_tensor[:,:,i])
        
        edge_tensor[:,:,i] = edge_image
        edge_features[i] = [
            global_stats['edge_density'],
            global_stats['total_edges'],
            global_stats['non_zero_ratio']
        ]
    
    return {
        'edge_tensor': edge_tensor,
        'edge_features': edge_features,
        'feature_names': ['edge_density', 'total_edges', 'non_zero_ratio'],
        'shape': edge_tensor.shape
    }