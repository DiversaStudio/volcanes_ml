import numpy as np
import torch

def prepare_labels(dataset):
    """
    Prepare target variables (labels) for volcanic status classification.
    
    :param dataset: Dictionary containing the processed dataset
    :return: Dictionary with encoded labels and mapping
    """
    # Obtener etiquetas crudas desde el metadata
    raw_labels = np.array(dataset['metadata']['labels'])
    
    # Definir el mapeo de etiquetas
    label_mapping = {
        'Despejado': 0,  # Clear
        'Nublado': 1,    # Cloudy
        'Emisiones': 2   # Emissions
    }
    
    # Convertir etiquetas de texto a etiquetas numéricas
    numeric_labels = np.array([label_mapping[label] for label in raw_labels])
    
    # Convertir etiquetas numéricas a tensor PyTorch de tipo LongTensor y aplicar one-hot encoding
    numeric_labels_tensor = torch.tensor(numeric_labels, dtype=torch.long)
    onehot_labels = torch.nn.functional.one_hot(numeric_labels_tensor, num_classes=len(label_mapping)).float()
    
    return {
        'numeric_labels': numeric_labels_tensor,   # Tensor con etiquetas numéricas
        'onehot_labels': onehot_labels,            # Tensor one-hot encoded
        'raw_labels': raw_labels,                  # Etiquetas originales en texto
        'label_mapping': label_mapping,            # Diccionario de mapeo
        'n_classes': len(label_mapping)
    }