# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project for volcanic activity classification using thermal imaging data from FLIR cameras. The system processes thermal images (FFF format) from Ecuadorian volcanoes (Cotopaxi and Reventador) and classifies them into four categories:
- **Despejado** (Clear/No activity)
- **Nublado** (Cloudy conditions)
- **Emisiones** (Emissions)
- **Flujo** (Lava Flow)

The project uses a multi-branch CNN architecture that processes three types of features in parallel:
1. Radiometrically corrected thermal images
2. Canny edge detection images
3. Multi-level temperature threshold tensors (low, medium, high, very_high)

## Data Pipeline Architecture

### 1. Data Loading (`src/data/loader.py`)
- `ThermalDataLoader` reads FLIR FFF files using the `flirpy` library
- Applies radiometric corrections using Planck's law and atmospheric transmission calculations
- Extracts metadata including timestamps, camera parameters (Planck R1, R2, B, F, O), and environmental conditions
- Processes images in batches to manage memory (default batch_size=500)
- Creates numpy tensors with shape `(height, width, n_samples)` where images are stacked along the third dimension

### 2. Preprocessing (`src/data/preprocessor.py`)
- `prepare_labels()` maps Spanish text labels to numeric values (0-3)
- Generates both numeric labels and one-hot encoded tensors for training

### 3. Feature Engineering
- **Edge Detection** (`src/features/edge_detection.py`): Applies Canny edge detection after histogram equalization. Returns binary edge images and statistics (edge_density, total_edges, non_zero_ratio)
- **Thermal Thresholds** (`src/features/thermal.py`): Creates binary threshold tensors at 4 temperature levels (0.1°C, 2.0°C, 5.0°C, 10.0°C)
- **Zonal Statistics** (`src/features/zonal.py`): Calculates mean, std, min, max, median for image regions

### 4. Dataset Structure (`src/data/dataset.py`)
- `ThermalDataset`: Custom PyTorch Dataset that loads preprocessed tensors
- `CustomTensorDataset`: Alternative dataset for loading saved tensor files
- Each sample contains: `corrected` (1 channel), `edge` (1 channel), `thresholds` (4 channels concatenated), and `label`
- Datasets are saved separately by split (train/val) with subdirectories for tensors, thresholds, and metadata

### 5. Model Architecture (`src/models/multibranch.py`)
- `MultiBranchModel`: Three parallel CNN branches that converge into fully connected layers
- Expected input dimensions: `[batch_size, 1, 240, 320]` for corrected/edge, `[batch_size, 4, 240, 320]` for thresholds
- Each branch uses progressive spatial reduction: Conv2d → BatchNorm → ReLU → Dropout2d/MaxPool2d → AdaptiveAvgPool2d
- Each branch outputs 64 features, concatenated to 192 features total before FC layers
- Configurable dropout rate (default 0.6)

### 6. Training (`src/models/trainer.py`)
- `train_epoch()` and `validate()` functions handle the training loop
- Moves all three input tensors (corrected, edge, thresholds) to device
- Uses tqdm for progress tracking

### 7. Prediction (`src/models/predictions.py`)
- `load_model()` loads checkpoint and returns model + label mapping
- `process_single_file()` handles end-to-end prediction on FFF files
- `process_directory()` batch processes entire directories and outputs CSV results
- Generates visualizations showing thermal image, edge detection, thresholds, and prediction confidence

## Data Directory Structure

```
data/
├── input/               # Raw FFF files organized by volcano and label
│   ├── Cotopaxi/
│   │   ├── Despejado/
│   │   ├── Emisiones/
│   │   └── Nublado/
│   └── Reventador/
│       ├── Despejado/
│       ├── Emisiones/
│       ├── Flujo/
│       └── Nublado/
├── processed/           # Saved PyTorch tensor datasets
│   ├── train/
│   │   ├── tensors/
│   │   │   ├── corrected.pt
│   │   │   ├── edge.pt
│   │   │   └── threshold/
│   │   │       ├── low.pt
│   │   │       ├── medium.pt
│   │   │       ├── high.pt
│   │   │       └── very_high.pt
│   │   └── metadata.pt
│   └── val/            # Same structure as train/
└── results/            # Model outputs, predictions, visualizations
```

## Development Workflow

### Jupyter Notebooks (Sequential Pipeline)
The project follows a numbered notebook workflow in `notebooks/`:

1. **00_VolcanoData.ipynb**: Exploratory data analysis, dataset statistics visualization
2. **01_DataPreprocessing.ipynb**: Load FFF files, apply radiometric corrections, create tensors
3. **02_FeatureEngineering.ipynb**: Generate edge detection and threshold features
4. **03_ModelTraining.ipynb**: Train the multi-branch model with train/val split
5. **04_Prediction.ipynb**: Run inference on new data

### Working with Notebooks
- Notebooks should be run in order as they build upon previous outputs
- Each notebook saves intermediate results to `data/processed/` or `data/results/`
- Notebooks import from `src/` modules for all processing functions

### Python Environment
- This project uses a virtual environment named `volcanesML` (visible in directory tree)
- Activate with: `volcanesML\Scripts\activate` (Windows) or `source volcanesML/bin/activate` (Unix)
- Dependencies are listed in `requirements.txt` (currently empty but project requires: torch, numpy, opencv-python, matplotlib, pandas, tqdm, scipy, flirpy)

### Working with FLIR FFF Files
- The `flirpy` library (included as subdirectory) handles FFF file I/O
- `flirpy` is a third-party library for FLIR thermal cameras - see `flirpy/README.md` for documentation
- Use `Fff(file_path)` to read files and `fff_reader.get_image()` to extract thermal data
- Metadata is accessed via `fff_reader.meta` and contains camera parameters needed for radiometric correction

## Memory Management

Large datasets require careful memory handling:
- Image tensors are `float32` by default
- Expected dimensions: 480×640 pixels per image
- Batch processing (500 images at a time) prevents memory overflow
- Use `save_large_dataset()` and `load_large_dataset()` to persist tensors to disk
- Consider reducing `limit_per_directory` in `ThermalDataLoader` for testing

## Model Training Notes

- Models are saved with checkpoints containing: `model_state_dict`, `optimizer_state_dict`, `epoch`, `loss`, `accuracy`
- Early stopping is implemented in `src/models/earlystop.py`
- Label mapping is fixed: {Despejado: 0, Nublado: 1, Emisiones: 2, Flujo: 3}
- Not all volcanoes have all classes (e.g., Cotopaxi has no Flujo examples in current data)
- Device selection: Use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

## Key Conventions

- All temperature values are in Celsius (converted from Kelvin internally)
- Timestamps are stored as Python datetime objects (UTC)
- Images use numpy arrays during processing, converted to PyTorch tensors for model input
- File naming convention for FFF files: `{VOLCANO}_{CAMERA}_{YYYYMMDD}_{HHMM}.fff`
- Labels are in Spanish as they correspond to official volcanic monitoring terminology
