import numpy as np

def calculate_zonal_statistics(image):
    """Calculate basic statistical metrics for an image region."""
    return {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'median': np.median(image)
    }
