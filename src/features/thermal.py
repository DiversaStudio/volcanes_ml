import numpy as np
from scipy.constants import c, h, k
import torch

def planck(wavelength, temperature):
    """Calculate spectral radiance using Planck's law."""
    return (2 * h * c**2) / (wavelength**5 * (np.exp((h * c) / (wavelength * k * temperature)) - 1))

def create_thermal_threshold_tensor(thermal_tensor, thresholds=None):
    """Create binary tensors for different temperature thresholds."""
    height, width, time = thermal_tensor.shape
    
    # Allow user to set thresholds manually
    if thresholds is None:
        thresholds = {
        'low': 0.1,      # Low temperature threshold
        'medium': 2.0,   # Medium temperature threshold
        'high': 5.0,     # High temperature threshold
        'very_high': 10.0  # Very high temperature threshold
        }
    
    threshold_tensors = {}
    device = thermal_tensor.device

    for name, temp in thresholds.items():
        threshold_tensor = (thermal_tensor > temp).type(torch.uint8) * 255
        threshold_tensors[name] = threshold_tensor.to(device)
    
    return {
        'tensors': threshold_tensors,
        'thresholds': thresholds
    }