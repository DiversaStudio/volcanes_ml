# VolcanesML: Volcanic Thermal Image Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine learning system for automated classification of volcanic activity using thermal imaging data from FLIR cameras. Developed in collaboration with the Geophysical Institute at Escuela Politécnica Nacional (IG-EPN) and Diversa.

## Overview

VolcanesML is a deep learning pipeline that processes thermal images (FFF format) from Ecuadorian volcanoes (Cotopaxi and Reventador) and classifies them into four categories of volcanic activity:

- **Despejado** (Clear/No activity)
- **Nublado** (Cloudy conditions)
- **Emisiones** (Emissions)
- **Flujo** (Lava flow)

The system uses a **multi-branch CNN architecture** that processes three types of features in parallel:

```
Input Image (FLIR FFF)
         |
         v
  ┌──────────────────┐
  │  Radiometric     │
  │  Correction      │
  └──────────────────┘
         |
    ┌────┴────┬──────────────┐
    |         |              |
    v         v              v
┌───────┐ ┌──────┐  ┌──────────────┐
│ Thermal│ │ Canny│  │   Thermal    │
│ Image │ │ Edge │  │  Thresholds  │
│(1 ch) │ │(1 ch)│  │  (4 levels)  │
└───────┘ └──────┘  └──────────────┘
    |         |              |
    └────┬────┴──────────────┘
         |
    Multi-Branch CNN
         |
         v
   Classification
  (4-class output)
```

**Performance**: Achieves **94.6% validation accuracy** on full dataset (10,560 thermal images).

---

## ⚠️ IMPORTANT: Data Directory Structure

This repository uses **two separate data directories**:

### `data-sample/` ✅ (Included in Repository)

- **Size**: ~30-50 MB (~35-40 FFF files)
- **Purpose**: Quick testing, code demonstration, pipeline validation
- **Sampling**: 0.05-0.1% of full dataset (~5 files per class)
- **Included in git**: ✅ Yes
- **Use for**: Testing code, learning pipeline, quick iteration
- **Performance**: ⚠️ Low accuracy (40-60%) due to limited data

### `data/` ❌ (NOT Included - Too Large)

- **Size**: ~50-60 GB (10,560 FFF files)
- **Purpose**: Production model training, research, publication
- **Sampling**: 100% (complete dataset)
- **Included in git**: ❌ No (too large, use `.gitignore`)
- **Use for**: Production models, achieving 94.6% accuracy
- **Obtaining**: Contact IG-EPN (see Data Access section below)

**Default Configuration**: The project is configured to use `data/` by default. If you're using sample data, update `config/config.yaml`:

```yaml
paths:
  input: "data-sample/input"  # Change from "data/input"
  processed: "data-sample/processed"  # Change from "data/processed"
```

**⚠️ Warning**: Training with `data-sample/` will result in poor model performance. Use full `data/` directory for production models (94.6% accuracy).

See `data-sample/README.md` for detailed information about sample vs. full dataset.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU-only mode supported)
- ~2GB disk space for dependencies
- Additional space for data (sample data: ~30MB, full dataset: ~50GB)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/volcanesml.git
cd volcanesML
```

2. **Create a virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify setup:**
```bash
python scripts/verify_setup.py
```

## Quick Start

The project follows a sequential notebook-based workflow:

1. **Exploratory Data Analysis** (`notebooks/00_VolcanoData.ipynb`)
   - Dataset statistics and visualization
   - Class distribution analysis

2. **Data Preprocessing** (`notebooks/01_DataPreprocessing.ipynb`)
   - Load FLIR FFF files
   - Apply radiometric corrections using Planck's law
   - Create preprocessed tensors

3. **Feature Engineering** (`notebooks/02_FeatureEngineering.ipynb`)
   - Generate Canny edge detection features
   - Create multi-level thermal threshold tensors

4. **Model Training** (`notebooks/03_ModelTraining.ipynb`)
   - Train multi-branch CNN (3 parallel branches)
   - Validation with early stopping
   - Save best model checkpoint

5. **Prediction** (`notebooks/04_Prediction.ipynb`)
   - Run inference on new thermal images
   - Generate visualizations and predictions

Each notebook imports from the `src/` modules and uses centralized configuration (`config/config.yaml`).

## Data Format

### FLIR FFF Files

The project processes FLIR thermal imaging files with the following specifications:

- **Format**: FFF (FLIR proprietary format)
- **Resolution**: 480×640 pixels (or 240×320 for some cameras)
- **File size**: ~600KB per image
- **Naming convention**: `{VOLCANO}_{CAMERA}_{YYYYMMDD}_{HHMM}.fff`
- **Metadata**: Includes Planck parameters (R1, R2, B, F, O), atmospheric conditions, timestamps

### Radiometric Correction

Raw thermal data is converted to calibrated temperature values using:

```
T(°C) = B / ln(R1 / (R2 * (S + O)) + F) - 273.15
```

Where:
- S: Raw sensor signal
- R1, R2, B, F, O: Planck calibration parameters (from camera metadata)
- Atmospheric transmission and emissivity corrections applied

### Dataset Structure

#### Full Dataset Structure (`data/` - NOT in repository)

```
data/                          # ❌ NOT INCLUDED (too large, ~50-60GB)
├── input/                     # Full raw FFF files (10,560 images)
│   ├── Cotopaxi/
│   │   ├── Despejado/        # ~2,800 images
│   │   ├── Emisiones/        # ~1,900 images
│   │   └── Nublado/          # ~2,300 images
│   └── Reventador/
│       ├── Despejado/        # ~1,200 images
│       ├── Emisiones/        # ~1,100 images
│       ├── Flujo/            # ~436 images
│       └── Nublado/          # ~800 images
├── processed/                 # Generated tensors (~43GB)
│   ├── train/
│   ├── val/
│   └── models/
└── results/                   # Predictions and visualizations
```

#### Sample Dataset Structure (`data-sample/` - Included in repository)

```
data-sample/                   # ✅ INCLUDED (~30-50MB)
├── input/                     # Sample FFF files (~35-40 images)
│   ├── Cotopaxi/
│   │   ├── Despejado/        # ~5 sample files
│   │   ├── Emisiones/        # ~5 sample files
│   │   └── Nublado/          # ~5 sample files
│   └── Reventador/
│       ├── Despejado/        # ~5 sample files
│       ├── Emisiones/        # ~5 sample files
│       ├── Flujo/            # ~5 sample files (unique to Reventador)
│       └── Nublado/          # ~5 sample files
├── processed/                 # Generated tensors (gitignored)
├── test/                      # Sample test files
└── results/                   # Sample results (gitignored)
```

**⚠️ Important**:
- `data/` is **NOT included** in git (too large, ~50-60GB)
- `data-sample/` is **included** in git (~30-50MB)
- Use `data-sample/` for quick testing
- Obtain full `data/` from IG-EPN for production training
- See `data-sample/README.md` for detailed comparison

## Usage

### Training a Model

```python
from src.data.loader import ThermalDataLoader
from src.data.dataset import ThermalDataset
from src.models.multibranch import MultiBranchModel
from src.models.trainer import train_epoch, validate
from src.config import get_config

# Load configuration
config = get_config()

# Load and preprocess data
loader = ThermalDataLoader(
    base_dir=config.paths['input'],
    batch_size=config.data['batch_size']
)
data = loader.load_all_data()

# Create dataset
dataset = ThermalDataset(data, config)

# Initialize model
model = MultiBranchModel(
    n_classes=config.model['n_classes'],
    dropout_rate=config.model['dropout_rate']
)

# Train
# See notebooks/03_ModelTraining.ipynb for complete training loop
```

### Making Predictions

```python
from src.models.predictions import process_single_file

# Process a single FFF file
result = process_single_file(
    fff_path='path/to/thermal_image.fff',
    model_path='data/processed/models/best_model.pt',
    output_dir='data/results/',
    save_visualization=True
)

print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Processing

```python
from src.models.predictions import process_directory

# Process entire directory
results_df = process_directory(
    input_dir='data/test/DiarioRaw/',
    model_path='data/processed/models/best_model.pt',
    output_dir='data/results/'
)

# Results saved to CSV with predictions and confidences
```

## Project Structure

```
volcanesML/
├── config/
│   └── config.yaml           # Centralized configuration
├── data/
│   ├── input/                # Raw thermal images (FFF)
│   ├── processed/            # Preprocessed tensors (gitignored)
│   ├── test/                 # Test data
│   └── results/              # Predictions and outputs
├── figures/                  # Visualizations
├── notebooks/                # Jupyter notebooks (00-04)
│   ├── 00_VolcanoData.ipynb
│   ├── 01_DataPreprocessing.ipynb
│   ├── 02_FeatureEngineering.ipynb
│   ├── 03_ModelTraining.ipynb
│   └── 04_Prediction.ipynb
├── scripts/
│   └── verify_setup.py       # Environment verification
├── src/                      # Source code
│   ├── config.py             # Configuration loader
│   ├── data/
│   │   ├── loader.py         # FFF file loading
│   │   ├── preprocessor.py   # Data preprocessing
│   │   └── dataset.py        # PyTorch datasets
│   ├── features/
│   │   ├── edge_detection.py # Canny edge features
│   │   ├── thermal.py        # Thermal thresholds
│   │   └── zonal.py          # Statistical features
│   └── models/
│       ├── multibranch.py    # CNN architecture
│       ├── trainer.py        # Training utilities
│       ├── predictions.py    # Inference pipeline
│       └── earlystop.py      # Early stopping
├── .gitignore
├── CLAUDE.md                 # AI assistant instructions
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Model Architecture

The `MultiBranchModel` consists of three parallel CNN branches:

1. **Corrected Branch**: Processes radiometrically corrected thermal images (1 channel)
   - Conv2d(1→32) → BatchNorm → ReLU → Dropout2d
   - Conv2d(32→64) → BatchNorm → ReLU → MaxPool2d
   - AdaptiveAvgPool2d → Flatten → 64 features

2. **Edge Branch**: Processes Canny edge detection images (1 channel)
   - Same architecture as Corrected Branch
   - 64 features output

3. **Threshold Branch**: Processes multi-level thermal thresholds (4 channels)
   - Conv2d(4→32) → BatchNorm → ReLU → Dropout2d
   - Conv2d(32→64) → BatchNorm → ReLU → MaxPool2d
   - AdaptiveAvgPool2d → Flatten → 64 features

**Feature Fusion**: Branches concatenate to 192 features total

**Classification Head**:
- FC(192→128) → BatchNorm → ReLU → Dropout(0.6)
- FC(128→64) → BatchNorm → ReLU → Dropout(0.6)
- FC(64→4) output

**Expected Input Dimensions**:
- Corrected: `[batch_size, 1, 240, 320]`
- Edge: `[batch_size, 1, 240, 320]`
- Thresholds: `[batch_size, 4, 240, 320]`

## Configuration

All parameters are centralized in `config/config.yaml`:

- **Paths**: Data directories, model checkpoints
- **Hyperparameters**: Learning rate, batch size, epochs
- **Thresholds**: Volcano-specific thermal thresholds (Cotopaxi vs. Reventador)
- **Model Architecture**: Classes, dropout, channels

Modify `config.yaml` instead of hardcoding values in notebooks or scripts.

## Memory Management

Processing large thermal datasets requires careful memory handling:

- Image tensors use `float32` precision
- Batch processing (default: 500 images) prevents memory overflow
- Use `limit_per_directory` in config for testing with subsets
- Preprocessed tensors can be saved/loaded to avoid recomputation

**Estimated Memory Requirements**:
- Training: ~4-6GB GPU memory (batch_size=250)
- Full dataset processing: ~8-12GB RAM
- Preprocessed tensors: ~43GB disk space (full dataset)

## Reproducibility

To reproduce published results:

1. Use the same train/validation split (80/20)
2. Set random seed: `torch.manual_seed(42)`
3. Use configuration from `config/config.yaml`
4. Expected validation accuracy: **94.6% ± 2%**

## Troubleshooting

### Common Issues

**1. FLIR FFF files not loading**
- Ensure `flirpy` is installed: `pip install flirpy`
- Verify FFF files are not corrupted (check file size ~600KB)

**2. CUDA out of memory**
- Reduce `batch_size` in `config/config.yaml`
- Use CPU mode: set `device: "cpu"` in config

**3. NaN values in thermal data**
- Some FFF files may have invalid pixels
- Preprocessing handles NaN values automatically

**4. Module import errors**
- Ensure project root is in Python path
- Run notebooks with kernel in virtual environment

## Citation

If you use this code or dataset, please cite:

```bibtex
@software{volcanesml2025,
  title={VolcanesML: Volcanic Thermal Image Classification},
  author={Geophysical Institute, Escuela Politécnica Nacional},
  year={2025},
  url={https://github.com/your-org/volcanesml},
  note={Multi-branch CNN for automated volcanic activity monitoring}
}
```

## Data Access

### Sample Data (Included in Repository)

The `data-sample/` directory contains ~35-40 thermal images representing all volcano classes:

- **Size**: ~30-50 MB
- **Location**: `data-sample/input/`
- **Purpose**: Quick testing, code demonstration, learning
- **Included**: ✅ Yes (already in git repository)
- **Performance**: ⚠️ Low accuracy (40-60%) - insufficient for production

**Use this for**: Testing the pipeline, validating code changes, quick experimentation

### Full Dataset (Contact IG-EPN)

To obtain the **complete dataset** (10,560 thermal images, ~6.5GB):

**Contact**: Geophysical Institute - Escuela Politécnica Nacional (IG-EPN)
**Website**: https://www.igepn.edu.ec/
**Email**: svallejo@igepn.edu.ec
**Data**: Available upon request for research and educational purposes

**Full Dataset Includes**:
- 7,024 images from Cotopaxi (Despejado, Nublado, Emisiones)
- 3,536 images from Reventador (Despejado, Nublado, Emisiones, Flujo)
- Complete metadata and timestamps
- Production-quality training capability (94.6% accuracy)

**Use this for**: Production model training, research publications, achieving reported performance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Diversa** team for the entire model design and development (https://diversa.studio/)
- **Geophysical Institute (IG-EPN)** for providing thermal imaging data
- **FLIR Systems** for thermal camera technology
- **PyTorch** team for the deep learning framework

---

**Maintained by**: Geophysical Institute - EPN
**Status**: Production-ready (v1.0.0)
**Last Updated**: February 2025
