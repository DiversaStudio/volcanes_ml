import numpy as np

def segment_image(image):
    """Divide an image into 9 segments (3x3 grid)."""
    h, w = image.shape
    segments = []
    for i in range(3):
        for j in range(3):
            segment = image[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            segments.append(segment)
    return segments

def calculate_zonal_statistics(image):
    """Calculate basic statistical metrics for an image region."""
    return {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'median': np.median(image)
    }

def format_zonal_stats_for_cnn(dataset):
    """Format zonal statistics into a tensor suitable for CNN input."""
    segment_stats = dataset['statistics']['corrected_segments']
    n_images = len(segment_stats)
    
    zonal_features = np.zeros((n_images, 45))
    
    for i, image_segments in enumerate(segment_stats):
        for j, segment in enumerate(image_segments):
            features = [
                segment['mean'],
                segment['std'],
                segment['min'],
                segment['max'],
                segment['median']
            ]
            zonal_features[i, j*5:(j+1)*5] = features
    
    return zonal_features