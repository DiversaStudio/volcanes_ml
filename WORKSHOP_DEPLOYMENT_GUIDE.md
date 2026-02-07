# VolcanesML: Technical Workshop & Deployment Guide

**Target Audience**: Systems Engineers at IG-EPN
**Objective**: Deploy production-ready volcanic activity classification system
**Last Updated**: February 2026

---

## Table of Contents

1. [Workshop Overview](#1-workshop-overview)
2. [System Architecture](#2-system-architecture)
3. [Technical Prerequisites](#3-technical-prerequisites)
4. [Installation & Environment Setup](#4-installation--environment-setup)
5. [Understanding the Pipeline](#5-understanding-the-pipeline)
6. [Data Processing Workflow](#6-data-processing-workflow)
7. [Model Architecture Deep Dive](#7-model-architecture-deep-dive)
8. [Training & Validation](#8-training--validation)
9. [Production Deployment](#9-production-deployment)
10. [Operational Guidelines](#10-operational-guidelines)
11. [Troubleshooting & Maintenance](#11-troubleshooting--maintenance)
12. [Hands-On Exercises](#12-hands-on-exercises)

---

## 1. Workshop Overview

### What This System Does

**VolcanesML** is an automated volcanic activity classification system that:
- Processes thermal images from FLIR cameras monitoring Ecuadorian volcanoes
- Classifies volcanic states into 4 categories using deep learning
- Operates on thermal FFF files captured from Cotopaxi and Reventador volcanoes

### Classification Categories

| Spanish Label | English | Description |
|--------------|---------|-------------|
| **Despejado** | Clear | No visible volcanic activity, clear sky |
| **Nublado** | Cloudy | Cloud cover obscuring view, no thermal anomalies |
| **Emisiones** | Emissions | Gas/steam emissions, moderate thermal activity |
| **Flujo** | Lava Flow | Active lava flows (Reventador only) |

### Key Performance Metrics

- **Validation Accuracy**: 94.6%
- **Dataset Size**: 10,560 thermal images
- **Image Resolution**: 480×640 or 240×320 pixels
- **Processing Speed**: ~0.5-1 second per image (GPU)

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FLIR Camera Network                       │
│            (Cotopaxi & Reventador Monitoring)                │
└────────────────────────┬────────────────────────────────────┘
                         │ FFF Files
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Ingestion Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ FFF Reader   │  │ Radiometric  │  │   Metadata   │     │
│  │  (flirpy)    │→ │  Correction  │→ │  Extraction  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │ Corrected Thermal Data
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Feature Engineering Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Thermal    │  │  Canny Edge  │  │  Threshold   │     │
│  │    Image     │  │  Detection   │  │   Tensors    │     │
│  │  (1 channel) │  │  (1 channel) │  │ (4 channels) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │ Multi-channel Features
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-Branch CNN Model                          │
│                                                               │
│   Branch 1        Branch 2         Branch 3                  │
│  ┌────────┐     ┌────────┐      ┌────────┐                 │
│  │Thermal │     │  Edge  │      │Threshold│                 │
│  │  CNN   │     │  CNN   │      │  CNN    │                 │
│  │ (64f)  │     │ (64f)  │      │ (64f)   │                 │
│  └────┬───┘     └────┬───┘      └────┬────┘                 │
│       └──────────────┴───────────────┘                       │
│                      │ Concatenate (192 features)            │
│                      ▼                                        │
│              ┌──────────────┐                                │
│              │ FC Layers    │                                │
│              │ 192→128→64→4 │                                │
│              └──────────────┘                                │
└────────────────────────┬────────────────────────────────────┘
                         │ Classification
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output & Monitoring                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Prediction   │  │ Confidence   │  │Visualization │     │
│  │   (0-3)      │  │ Score (%)    │  │    (PNG)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Deep Learning** | PyTorch 2.0+ | Model architecture & training |
| **Thermal I/O** | flirpy | FLIR FFF file reading |
| **Image Processing** | OpenCV, NumPy | Edge detection, preprocessing |
| **Scientific Computing** | SciPy, pandas | Statistical analysis |
| **Visualization** | Matplotlib, Seaborn | Results visualization |
| **Development** | Jupyter, Python 3.8+ | Interactive development |

---

## 3. Technical Prerequisites

### Hardware Requirements

#### Minimum Specifications
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 100GB free space
  - Code: ~500MB
  - Full dataset: ~60GB
  - Processed tensors: ~43GB
- **GPU**: Optional (NVIDIA CUDA-capable)

#### Recommended for Production
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB
- **Storage**: 200GB SSD
- **GPU**: NVIDIA GPU with 6GB+ VRAM (e.g., RTX 3060, Tesla T4)
- **Network**: 100Mbps+ for data transfer

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **Python** | 3.8 - 3.11 | Runtime environment |
| **pip** | Latest | Package management |
| **CUDA** | 11.8+ (optional) | GPU acceleration |
| **Git** | 2.0+ | Version control |
| **Jupyter** | Latest | Interactive notebooks |

### Network Access

Ensure firewall allows:
- PyPI (pip packages)
- GitHub (repository access)
- Internal network (data transfer from FLIR cameras)

---

## 4. Installation & Environment Setup

### Step 1: Clone Repository

```bash
# Navigate to installation directory
cd /opt/volcanesml  # Linux
cd C:\Programs\volcanesml  # Windows

# Clone repository
git clone https://github.com/your-org/volcanesml.git
cd volcanesML
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (GPU version - recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch (CPU-only version)
# pip install torch torchvision

# Install all dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Run verification script
python scripts/verify_setup.py
```

**Expected Output:**
```
✓ Python version: 3.9.x
✓ PyTorch installed: 2.1.x
✓ CUDA available: True (GPU detected)
✓ All dependencies installed
✓ Directory structure valid
✓ Sample data accessible
```

### Step 5: Configure Data Paths

Edit `config/config.yaml`:

```yaml
paths:
  # For testing with sample data
  input: "data-sample/input"
  processed: "data-sample/processed"

  # For production with full dataset
  # input: "data/input"
  # processed: "data/processed"

  results: "data/results"
  models: "data/processed/models"
```

---

## 5. Understanding the Pipeline

### Pipeline Stages Overview

```
Stage 1: Data Loading (loader.py)
   ↓
Stage 2: Radiometric Correction (loader.py)
   ↓
Stage 3: Feature Engineering (features/)
   ↓
Stage 4: Dataset Preparation (dataset.py)
   ↓
Stage 5: Model Training (trainer.py)
   ↓
Stage 6: Prediction (predictions.py)
```

### Stage 1: Data Loading

**File**: [`src/data/loader.py`](src/data/loader.py)

```python
from src.data.loader import ThermalDataLoader

# Initialize loader
loader = ThermalDataLoader(
    base_dir="data/input",
    batch_size=500,        # Process 500 images at a time
    limit_per_directory=None  # None = load all files
)

# Load all FFF files
data = loader.load_all_data()
# Returns: dict with 'corrected', 'metadata', 'labels'
```

**What happens:**
1. Scans directory structure for FFF files
2. Reads thermal data using `flirpy.Fff()`
3. Extracts camera parameters (R1, R2, B, F, O)
4. Organizes by volcano and label

### Stage 2: Radiometric Correction

**Physics Formula** (Planck's Law):

```
T(°C) = B / ln(R1 / (R2 * (S + O)) + F) - 273.15
```

Where:
- **S**: Raw sensor signal (counts)
- **R1, R2**: Planck radiation constants
- **B**: Temperature constant (Kelvin)
- **F, O**: Calibration offsets
- **T**: Temperature in Celsius

**Code implementation:**
```python
def apply_radiometric_correction(raw_data, planck_params):
    """
    raw_data: numpy array (height, width)
    planck_params: dict with R1, R2, B, F, O
    """
    S = raw_data.astype(np.float32)
    R1 = planck_params['R1']
    R2 = planck_params['R2']
    B = planck_params['B']
    F = planck_params['F']
    O = planck_params['O']

    # Apply Planck's law
    temp_kelvin = B / np.log(R1 / (R2 * (S + O)) + F)
    temp_celsius = temp_kelvin - 273.15

    return temp_celsius
```

### Stage 3: Feature Engineering

**File**: [`src/features/`](src/features/)

#### 3.1 Edge Detection ([`edge_detection.py`](src/features/edge_detection.py))

```python
from src.features.edge_detection import apply_canny_edge_batch

# Apply Canny edge detection
edges = apply_canny_edge_batch(
    thermal_images,
    low_threshold=50,
    high_threshold=150
)
# Output: Binary edge images (0 or 255)
```

**Technical Details:**
- Histogram equalization for contrast enhancement
- Gaussian blur (kernel=5) for noise reduction
- Canny edge detection with dual thresholds
- Returns binary edge maps + statistics

#### 3.2 Thermal Thresholds ([`thermal.py`](src/features/thermal.py))

```python
from src.features.thermal import create_threshold_tensors

# Create 4-level threshold tensors
thresholds = create_threshold_tensors(
    thermal_images,
    thresholds=[0.1, 2.0, 5.0, 10.0]  # °C above ambient
)
# Output: 4 binary channels [low, medium, high, very_high]
```

**Threshold Levels:**
1. **Low** (0.1°C): Detects subtle thermal anomalies
2. **Medium** (2.0°C): Moderate emissions
3. **High** (5.0°C): Strong thermal activity
4. **Very High** (10.0°C): Extreme heat (lava flows)

### Stage 4: Dataset Preparation

**File**: [`src/data/dataset.py`](src/data/dataset.py)

```python
from src.data.dataset import ThermalDataset
from torch.utils.data import DataLoader

# Create PyTorch dataset
dataset = ThermalDataset(
    corrected=corrected_images,
    edges=edge_images,
    thresholds=threshold_tensors,
    labels=numeric_labels
)

# Create data loader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

**Data Augmentation** (optional):
- Random horizontal flips
- Random rotations (±10°)
- Brightness adjustments

---

## 6. Data Processing Workflow

### Complete Processing Example

**Notebook**: [`notebooks/01_DataPreprocessing.ipynb`](notebooks/01_DataPreprocessing.ipynb)

```python
import torch
import numpy as np
from src.data.loader import ThermalDataLoader
from src.data.preprocessor import prepare_labels
from src.features.edge_detection import apply_canny_edge_batch
from src.features.thermal import create_threshold_tensors

# Step 1: Load data
loader = ThermalDataLoader(base_dir="data/input", batch_size=500)
data = loader.load_all_data()

# Step 2: Extract components
corrected = data['corrected']  # Shape: (H, W, N) where N = num images
labels_text = data['labels']    # List of Spanish labels
metadata = data['metadata']     # List of dicts with timestamps, camera params

# Step 3: Prepare labels
labels_numeric, labels_onehot = prepare_labels(labels_text)
# labels_numeric: [0, 1, 2, 3]
# labels_onehot: [[1,0,0,0], [0,1,0,0], ...]

# Step 4: Feature engineering
print("Generating edge features...")
edges = apply_canny_edge_batch(corrected)

print("Creating threshold tensors...")
thresholds = create_threshold_tensors(corrected)

# Step 5: Save processed data
print("Saving tensors...")
torch.save({
    'corrected': torch.tensor(corrected),
    'edges': torch.tensor(edges),
    'thresholds': torch.tensor(thresholds),
    'labels': torch.tensor(labels_numeric),
    'metadata': metadata
}, 'data/processed/train/processed_data.pt')

print("✓ Processing complete")
```

### Memory Management Strategies

**For Large Datasets (10K+ images):**

```python
# Process in batches to avoid memory overflow
def process_large_dataset(base_dir, output_dir, batch_size=500):
    loader = ThermalDataLoader(base_dir, batch_size=batch_size)

    # Process each batch separately
    for batch_idx, batch_data in enumerate(loader.load_batches()):
        print(f"Processing batch {batch_idx+1}...")

        # Extract batch
        corrected_batch = batch_data['corrected']

        # Process features
        edges_batch = apply_canny_edge_batch(corrected_batch)
        thresholds_batch = create_threshold_tensors(corrected_batch)

        # Save batch
        torch.save({
            'corrected': torch.tensor(corrected_batch),
            'edges': torch.tensor(edges_batch),
            'thresholds': torch.tensor(thresholds_batch),
            'labels': batch_data['labels']
        }, f'{output_dir}/batch_{batch_idx}.pt')

        # Clear memory
        del corrected_batch, edges_batch, thresholds_batch
        torch.cuda.empty_cache()
```

---

## 7. Model Architecture Deep Dive

### MultiBranchModel Class

**File**: [`src/models/multibranch.py`](src/models/multibranch.py)

```python
from src.models.multibranch import MultiBranchModel

# Initialize model
model = MultiBranchModel(
    n_classes=4,
    dropout_rate=0.6
)

# Model summary
print(model)
```

### Branch Architecture Details

#### Branch 1: Corrected Thermal Branch

```python
# Input: [batch_size, 1, 240, 320]
self.corrected_branch = nn.Sequential(
    # First conv block
    nn.Conv2d(1, 32, kernel_size=3, padding=1),     # → [B, 32, 240, 320]
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Dropout2d(0.3),

    # Second conv block
    nn.Conv2d(32, 64, kernel_size=3, padding=1),    # → [B, 64, 240, 320]
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),                                 # → [B, 64, 120, 160]

    # Global pooling
    nn.AdaptiveAvgPool2d((1, 1)),                   # → [B, 64, 1, 1]
    nn.Flatten()                                     # → [B, 64]
)
```

#### Branch 2: Edge Detection Branch

```python
# Input: [batch_size, 1, 240, 320]
# Identical architecture to Branch 1
self.edge_branch = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Dropout2d(0.3),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)
```

#### Branch 3: Threshold Branch

```python
# Input: [batch_size, 4, 240, 320]  # 4 threshold channels
self.threshold_branch = nn.Sequential(
    # First conv block (4 input channels instead of 1)
    nn.Conv2d(4, 32, kernel_size=3, padding=1),     # → [B, 32, 240, 320]
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.Dropout2d(0.3),

    # Second conv block
    nn.Conv2d(32, 64, kernel_size=3, padding=1),    # → [B, 64, 240, 320]
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),                                 # → [B, 64, 120, 160]

    # Global pooling
    nn.AdaptiveAvgPool2d((1, 1)),                   # → [B, 64, 1, 1]
    nn.Flatten()                                     # → [B, 64]
)
```

### Feature Fusion & Classification

```python
# Concatenate branch outputs: 64 + 64 + 64 = 192 features
# Input: [batch_size, 192]

self.classifier = nn.Sequential(
    # First FC layer
    nn.Linear(192, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.6),

    # Second FC layer
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.6),

    # Output layer
    nn.Linear(64, 4)  # 4 classes
)
```

### Forward Pass

```python
def forward(self, corrected, edge, thresholds):
    """
    corrected: [batch, 1, H, W]
    edge: [batch, 1, H, W]
    thresholds: [batch, 4, H, W]
    """
    # Process each branch
    feat_corrected = self.corrected_branch(corrected)    # [batch, 64]
    feat_edge = self.edge_branch(edge)                    # [batch, 64]
    feat_threshold = self.threshold_branch(thresholds)   # [batch, 64]

    # Concatenate features
    combined = torch.cat([
        feat_corrected,
        feat_edge,
        feat_threshold
    ], dim=1)  # [batch, 192]

    # Classification
    output = self.classifier(combined)  # [batch, 4]

    return output
```

### Model Statistics

```
Total Parameters: ~2.1M
Trainable Parameters: ~2.1M
Memory Footprint: ~8.4 MB (float32)

Per-Branch Parameters:
- Corrected Branch: ~650K
- Edge Branch: ~650K
- Threshold Branch: ~680K
- Classifier: ~120K
```

---

## 8. Training & Validation

### Training Configuration

**File**: [`src/models/trainer.py`](src/models/trainer.py)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.multibranch import MultiBranchModel
from src.models.trainer import train_epoch, validate
from src.models.earlystop import EarlyStopping

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
model = MultiBranchModel(n_classes=4, dropout_rate=0.6)
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5  # L2 regularization
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# Early stopping
early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    path='data/processed/models/best_model.pt'
)
```

### Training Loop

**Notebook**: [`notebooks/03_ModelTraining.ipynb`](notebooks/03_ModelTraining.ipynb)

```python
from tqdm import tqdm

# Training hyperparameters
num_epochs = 100
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )

    # Validation phase
    val_loss, val_acc = validate(
        model, val_loader, criterion, device
    )

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Print metrics
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2%}")

    # Early stopping check
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

print("Training complete!")
```

### train_epoch Function

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for corrected, edge, thresholds, labels in pbar:
        # Move to device
        corrected = corrected.to(device)
        edge = edge.to(device)
        thresholds = thresholds.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(corrected, edge, thresholds)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy
```

### validate Function

```python
def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for corrected, edge, thresholds, labels in dataloader:
            # Move to device
            corrected = corrected.to(device)
            edge = edge.to(device)
            thresholds = thresholds.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(corrected, edge, thresholds)
            loss = criterion(outputs, labels)

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy
```

### Training Best Practices

1. **Monitor GPU usage**: `nvidia-smi -l 1`
2. **Save checkpoints regularly**: Every 5-10 epochs
3. **Use TensorBoard** (optional):
   ```python
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter('runs/experiment_1')
   writer.add_scalar('Loss/train', train_loss, epoch)
   writer.add_scalar('Accuracy/val', val_acc, epoch)
   ```
4. **Track learning curves**: Plot loss/accuracy vs epochs

---

## 9. Production Deployment

### Making Predictions on New Data

**File**: [`src/models/predictions.py`](src/models/predictions.py)

#### Single File Prediction

```python
from src.models.predictions import process_single_file

# Process one FFF file
result = process_single_file(
    fff_path='data/test/Cotopaxi_FFF_20250206_0800.fff',
    model_path='data/processed/models/best_model.pt',
    output_dir='data/results/',
    save_visualization=True
)

# Display results
print(f"Predicted Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All Probabilities:")
for label, prob in result['probabilities'].items():
    print(f"  {label}: {prob:.2%}")
```

**Output:**
```
Predicted Class: Emisiones
Confidence: 96.3%
All Probabilities:
  Despejado: 1.2%
  Nublado: 2.1%
  Emisiones: 96.3%
  Flujo: 0.4%
```

#### Batch Directory Processing

```python
from src.models.predictions import process_directory

# Process entire directory
results_df = process_directory(
    input_dir='data/test/DiarioRaw/',
    model_path='data/processed/models/best_model.pt',
    output_dir='data/results/',
    save_visualizations=True  # Save PNG for each prediction
)

# Results saved to CSV
# data/results/predictions_20250206_140523.csv
```

**CSV Output Format:**
```csv
filename,predicted_class,confidence,despejado_prob,nublado_prob,emisiones_prob,flujo_prob,timestamp
Cotopaxi_FFF_20250206_0800.fff,Emisiones,0.963,0.012,0.021,0.963,0.004,2025-02-06 08:00:00
Reventador_FFF_20250206_0815.fff,Flujo,0.891,0.003,0.015,0.091,0.891,2025-02-06 08:15:00
```

### Production Inference Pipeline

```python
import os
import torch
from pathlib import Path
from src.models.predictions import load_model, process_single_file

class VolcanoClassifier:
    """Production classifier wrapper"""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model, self.label_map = load_model(model_path, self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def predict(self, fff_path):
        """Predict single file"""
        result = process_single_file(
            fff_path=fff_path,
            model=self.model,
            label_map=self.label_map,
            device=self.device,
            save_visualization=False  # Faster without viz
        )
        return result

    def predict_batch(self, input_dir, output_csv):
        """Predict entire directory"""
        results = []

        for fff_file in Path(input_dir).glob('*.fff'):
            result = self.predict(str(fff_file))
            results.append({
                'filename': fff_file.name,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence'],
                **result['probabilities']
            })

        # Save to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        return df

# Usage
classifier = VolcanoClassifier('data/processed/models/best_model.pt')
results = classifier.predict_batch(
    input_dir='/data/flir/daily_captures',
    output_csv='results/daily_report.csv'
)
```

### Automated Monitoring Script

```python
#!/usr/bin/env python3
"""
Production monitoring script
Runs classification on new FLIR images and generates alerts
"""

import time
import logging
from pathlib import Path
from datetime import datetime
from src.models.predictions import process_directory

# Setup logging
logging.basicConfig(
    filename='logs/classifier.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
INPUT_DIR = '/data/flir/incoming'
OUTPUT_DIR = '/data/results'
MODEL_PATH = 'models/best_model.pt'
ALERT_THRESHOLD = 0.90  # Alert if confidence > 90%

def monitor_directory(interval=300):  # 5 minutes
    """Monitor directory for new FFF files"""

    logging.info("Starting monitoring service...")

    while True:
        try:
            # Check for new files
            fff_files = list(Path(INPUT_DIR).glob('*.fff'))

            if fff_files:
                logging.info(f"Found {len(fff_files)} new files")

                # Process batch
                results = process_directory(
                    input_dir=INPUT_DIR,
                    model_path=MODEL_PATH,
                    output_dir=OUTPUT_DIR
                )

                # Check for high-confidence emissions/flows
                alerts = results[
                    (results['predicted_class'].isin(['Emisiones', 'Flujo'])) &
                    (results['confidence'] > ALERT_THRESHOLD)
                ]

                if not alerts.empty:
                    logging.warning(f"ALERT: {len(alerts)} high-confidence volcanic activity detected!")
                    # Send alert (email, SMS, etc.)
                    send_alert(alerts)

                # Move processed files
                for fff_file in fff_files:
                    fff_file.rename(Path('/data/flir/processed') / fff_file.name)

                logging.info("Batch processing complete")

        except Exception as e:
            logging.error(f"Error during monitoring: {e}")

        # Wait before next check
        time.sleep(interval)

if __name__ == '__main__':
    monitor_directory()
```

### Systemd Service (Linux)

Create `/etc/systemd/system/volcano-classifier.service`:

```ini
[Unit]
Description=Volcano Thermal Classifier Service
After=network.target

[Service]
Type=simple
User=volcanesml
WorkingDirectory=/opt/volcanesml
ExecStart=/opt/volcanesml/venv/bin/python scripts/monitor_service.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable volcano-classifier
sudo systemctl start volcano-classifier
sudo systemctl status volcano-classifier
```

---

## 10. Operational Guidelines

### Daily Operations Checklist

- [ ] Check service status: `systemctl status volcano-classifier`
- [ ] Review logs: `tail -f logs/classifier.log`
- [ ] Monitor GPU usage: `nvidia-smi`
- [ ] Check disk space: `df -h /data`
- [ ] Verify latest predictions: `ls -lth data/results/ | head`
- [ ] Review accuracy metrics: Check validation reports

### Weekly Maintenance

- [ ] Backup model checkpoints
- [ ] Archive old predictions (>30 days)
- [ ] Review false positives/negatives
- [ ] Check system resource usage trends
- [ ] Update dependencies if needed

### Performance Monitoring

**Key Metrics to Track:**

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Inference Time | <1s per image | >2s |
| GPU Memory Usage | <80% | >90% |
| Prediction Confidence | >85% avg | <70% |
| Processing Queue | <10 pending | >50 |
| Service Uptime | 99.9% | <95% |

### Logging Strategy

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/classifier.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('VolcanoClassifier')

# Log predictions
logger.info(f"Processed {filename}: {predicted_class} ({confidence:.2%})")

# Log errors
logger.error(f"Failed to process {filename}: {error}")

# Log warnings
logger.warning(f"Low confidence prediction: {confidence:.2%}")
```

---

## 11. Troubleshooting & Maintenance

### Common Issues & Solutions

#### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**
1. Reduce batch size:
   ```python
   # In config/config.yaml
   training:
     batch_size: 16  # Reduce from 32
   ```

2. Clear GPU cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. Use gradient accumulation:
   ```python
   accumulation_steps = 4
   for i, (corrected, edge, thresholds, labels) in enumerate(train_loader):
       outputs = model(corrected, edge, thresholds)
       loss = criterion(outputs, labels) / accumulation_steps
       loss.backward()

       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

#### Issue 2: FFF Files Not Loading

**Symptoms:**
```
Error: Unable to read FFF file
```

**Diagnosis:**
```python
from flirpy.io.fff import Fff

try:
    fff = Fff('problematic_file.fff')
    image = fff.get_image()
    print("File OK")
except Exception as e:
    print(f"Error: {e}")
    # Check file size
    import os
    print(f"File size: {os.path.getsize('problematic_file.fff')} bytes")
```

**Solutions:**
- Verify file integrity (should be ~600KB)
- Check file permissions
- Ensure `flirpy` installed: `pip install flirpy`
- Try re-downloading corrupted files

#### Issue 3: Low Prediction Accuracy

**Symptoms:**
- Validation accuracy <70%
- High confidence on wrong predictions

**Diagnosis:**
```python
# Check model performance per class
from sklearn.metrics import classification_report, confusion_matrix

y_true = []  # Actual labels
y_pred = []  # Predicted labels

# Generate predictions
for corrected, edge, thresholds, labels in val_loader:
    outputs = model(corrected, edge, thresholds)
    _, predicted = outputs.max(1)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(predicted.cpu().numpy())

# Classification report
print(classification_report(y_true, y_pred, target_names=[
    'Despejado', 'Nublado', 'Emisiones', 'Flujo'
]))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
```

**Solutions:**
1. Check data quality:
   - Verify labels are correct
   - Check for corrupted images
   - Ensure balanced classes

2. Adjust hyperparameters:
   ```python
   # Increase dropout to reduce overfitting
   model = MultiBranchModel(n_classes=4, dropout_rate=0.7)

   # Lower learning rate
   optimizer = optim.Adam(model.parameters(), lr=0.0005)
   ```

3. Use data augmentation:
   ```python
   from torchvision import transforms

   transform = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomRotation(10),
       transforms.RandomAdjustSharpness(2)
   ])
   ```

#### Issue 4: Service Crashes

**Symptoms:**
```
systemctl status volcano-classifier
● volcano-classifier.service - failed
```

**Check logs:**
```bash
journalctl -u volcano-classifier -n 100 --no-pager
```

**Solutions:**
1. Check Python errors in logs
2. Verify model file exists:
   ```bash
   ls -lh data/processed/models/best_model.pt
   ```
3. Test script manually:
   ```bash
   source venv/bin/activate
   python scripts/monitor_service.py
   ```
4. Check file permissions
5. Increase memory limits in systemd service

### Performance Optimization

#### GPU Optimization

```python
# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Use mixed precision training (faster, less memory)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for corrected, edge, thresholds, labels in train_loader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(corrected, edge, thresholds)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### CPU Optimization

```python
# Use multiple data loading workers
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,  # Increase based on CPU cores
    pin_memory=True  # Faster GPU transfer
)

# Enable MKL for Intel CPUs
import torch
torch.set_num_threads(8)
```

### Backup & Recovery

**Backup Script:**
```bash
#!/bin/bash
# backup.sh - Backup models and critical data

BACKUP_DIR="/backup/volcanesml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "${BACKUP_DIR}/${TIMESTAMP}"

# Backup models
cp -r data/processed/models "${BACKUP_DIR}/${TIMESTAMP}/"

# Backup configuration
cp config/config.yaml "${BACKUP_DIR}/${TIMESTAMP}/"

# Backup logs (last 7 days)
find logs/ -mtime -7 -exec cp {} "${BACKUP_DIR}/${TIMESTAMP}/" \;

# Compress
cd "${BACKUP_DIR}"
tar -czf "backup_${TIMESTAMP}.tar.gz" "${TIMESTAMP}"
rm -rf "${TIMESTAMP}"

# Keep only last 30 days
find "${BACKUP_DIR}" -name "backup_*.tar.gz" -mtime +30 -delete

echo "Backup complete: ${BACKUP_DIR}/backup_${TIMESTAMP}.tar.gz"
```

**Recovery:**
```bash
# Extract backup
tar -xzf backup_20250206_120000.tar.gz

# Restore model
cp backup_20250206_120000/models/best_model.pt data/processed/models/

# Restart service
sudo systemctl restart volcano-classifier
```

---

## 12. Hands-On Exercises

### Exercise 1: Environment Setup (15 min)

**Objective**: Install and verify the system

1. Clone repository
2. Create virtual environment
3. Install dependencies
4. Run verification script
5. Explore directory structure

**Success Criteria**: `verify_setup.py` passes all checks

---

### Exercise 2: Data Processing (30 min)

**Objective**: Process sample thermal data

1. Open `notebooks/01_DataPreprocessing.ipynb`
2. Load 10 sample FFF files
3. Apply radiometric correction
4. Generate edge features
5. Create threshold tensors
6. Save processed tensors

**Questions:**
- What is the shape of the corrected tensor?
- How many edge pixels detected on average?
- What temperature range do thresholds cover?

---

### Exercise 3: Model Inspection (20 min)

**Objective**: Understand model architecture

1. Load pretrained model
2. Print model summary
3. Count parameters per branch
4. Visualize feature maps (optional)

**Code:**
```python
from src.models.multibranch import MultiBranchModel

model = MultiBranchModel(n_classes=4, dropout_rate=0.6)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
```

---

### Exercise 4: Training on Sample Data (45 min)

**Objective**: Train model on small dataset

1. Open `notebooks/03_ModelTraining.ipynb`
2. Load processed sample data
3. Split train/val (80/20)
4. Train for 20 epochs
5. Plot training curves
6. Evaluate validation accuracy

**Expected Result**: ~60-70% accuracy (sample data is limited)

---

### Exercise 5: Making Predictions (20 min)

**Objective**: Run inference on test images

1. Open `notebooks/04_Prediction.ipynb`
2. Load trained model
3. Process 5 test FFF files
4. Generate visualizations
5. Interpret results

**Deliverable**: CSV with predictions and confidence scores

---

### Exercise 6: Production Deployment (30 min)

**Objective**: Set up automated monitoring

1. Create monitoring script
2. Configure file paths
3. Test batch processing
4. Set up logging
5. (Optional) Create systemd service

**Test:**
```bash
# Run monitoring script manually
python scripts/monitor_service.py

# Check logs
tail -f logs/classifier.log
```

---

## Appendix A: Configuration Reference

### config/config.yaml

```yaml
# Paths
paths:
  input: "data/input"
  processed: "data/processed"
  results: "data/results"
  models: "data/processed/models"

# Data loading
data:
  batch_size: 500
  limit_per_directory: null  # null = load all
  train_split: 0.8
  random_seed: 42

# Feature engineering
features:
  edge_detection:
    low_threshold: 50
    high_threshold: 150
  thermal_thresholds:
    cotopaxi: [0.1, 2.0, 5.0, 10.0]  # °C
    reventador: [0.1, 2.0, 5.0, 10.0]

# Model architecture
model:
  n_classes: 4
  dropout_rate: 0.6
  input_size: [240, 320]

# Training
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 1e-5
  early_stopping_patience: 10
  scheduler_patience: 5
  scheduler_factor: 0.5

# Inference
inference:
  device: "cuda"  # or "cpu"
  confidence_threshold: 0.7
  save_visualizations: true

# Monitoring
monitoring:
  check_interval: 300  # seconds
  alert_threshold: 0.90
  log_level: "INFO"
```

---

## Appendix B: Volcano-Specific Notes

### Cotopaxi Characteristics

- **Elevation**: 5,897m
- **Camera distance**: ~10-15 km
- **Classes**: Despejado, Nublado, Emisiones
- **Typical temperatures**: -5°C to 15°C (ambient), up to 100°C (emissions)

### Reventador Characteristics

- **Elevation**: 3,562m
- **Camera distance**: ~5-8 km
- **Classes**: Nublado, Emisiones, Flujo
- **Typical temperatures**: 5°C to 25°C (ambient), 200-800°C (lava flows)
- **Note**: More frequent eruptive activity

---

## Appendix C: Useful Commands

### System Monitoring

```bash
# GPU monitoring
watch -n 1 nvidia-smi

# Disk space
df -h /data

# Memory usage
free -h

# Process monitoring
htop

# Service logs
journalctl -u volcano-classifier -f
```

### PyTorch Debugging

```python
# Check CUDA
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Check model device
print(next(model.parameters()).device)

# Memory summary
print(torch.cuda.memory_summary())
```

---

## Appendix D: Contact & Support

**Technical Support:**
- Email: svallejo@igepn.edu.ec

**Documentation:**
- README: [README.md](README.md)

**Contributors:**
- **Diversa**: Model architecture & development
- **IG-EPN**: Data collection & volcanic expertise

---

Last updated: February 2026
Version: 1.0.0

