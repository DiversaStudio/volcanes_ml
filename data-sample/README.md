# Sample Data Directory

⚠️ **IMPORTANT: This directory contains a small sample (~0.05-0.1%) of the full dataset for demonstration and testing purposes.**

---

## Purpose

The `data-sample/` directory provides:
- ✅ **Quick testing** - Run notebooks and scripts without downloading the full dataset
- ✅ **GitHub distribution** - Lightweight sample included in repository (<50MB)
- ✅ **Code demonstration** - Shows expected data structure and formats
- ✅ **Pipeline validation** - Verify the complete workflow works

---

## Directory Structure

```
data-sample/
├── input/                    # Sample raw FFF files
│   ├── Cotopaxi/
│   │   ├── Despejado/       # ~5 sample files
│   │   ├── Emisiones/       # ~5 sample files
│   │   └── Nublado/         # ~5 sample files
│   └── Reventador/
│       ├── Despejado/       # ~5 sample files
│       ├── Emisiones/       # ~5 sample files
│       ├── Flujo/           # ~5 sample files
│       └── Nublado/         # ~5 sample files
├── processed/                # Generated during preprocessing (gitignored)
├── test/                     # Sample test files
├── results/                  # Sample results (gitignored)
└── README.md                 # This file
```

---

## Sample vs Full Dataset

### Sample Dataset (data-sample/)

**Location**: `data-sample/` (included in git repository)

**Size**: ~30-50 MB (~35-40 FFF files)

**Sampling Rate**: 0.05-0.1% of full dataset

**Use Cases**:
- Quick testing and development
- Validating code changes
- Learning the pipeline
- Demonstrating functionality

**Files per Class**:
- Cotopaxi/Despejado: ~5 files (from 2,800)
- Cotopaxi/Emisiones: ~5 files (from 1,900)
- Cotopaxi/Nublado: ~5 files (from 2,300)
- Reventador/Despejado: ~5 files (from 1,200)
- Reventador/Emisiones: ~5 files (from 1,100)
- Reventador/Flujo: ~5 files (from 436)
- Reventador/Nublado: ~5 files (from 800)

### Full Dataset (data/)

**Location**: `data/` (NOT in git, use locally)

**Size**: ~50-60 GB (10,560 FFF files)

**Sampling Rate**: 100% (complete dataset)

**Use Cases**:
- Training production models
- Full validation and testing
- Research and publication
- Achieving best performance (94.6% accuracy)

**Obtaining Full Dataset**:
Contact: Geophysical Institute - Escuela Politécnica Nacional (IG-EPN)
Website: https://www.igepn.edu.ec/

---

## How Sample Data Was Created

The sample data was generated using `scripts/sample_data.py`:

```bash
# Generate sample with 5 files per directory
python scripts/sample_data.py --fixed-count 5

# Or generate sample with 0.05% sampling
python scripts/sample_data.py --percentage 0.05 --min-files 5
```

**Sampling Strategy**:
- **Reproducible**: Uses fixed random seed (42)
- **Evenly distributed**: Samples across date ranges
- **Representative**: Includes diverse conditions
- **Validated**: Only includes files without NaN values

---

## Using Sample Data

### Configuration

The project uses `config/config.yaml` for paths. By default, it points to `data/`:

```yaml
paths:
  input: "data/input"
  processed: "data/processed"
```

**To use sample data**, update your config:

```yaml
paths:
  input: "data-sample/input"
  processed: "data-sample/processed"
```

### Running Notebooks with Sample Data

1. **Update config.yaml** to point to `data-sample/`:
   ```yaml
   paths:
     input: "data-sample/input"
     processed: "data-sample/processed"
     results: "data-sample/results"
   ```

2. **Run preprocessing**:
   ```bash
   jupyter notebook notebooks/01_DataPreprocessing.ipynb
   ```

3. **Process will be much faster** (seconds vs minutes)

4. **Switch back to full data** when ready for production training

### Expected Results with Sample Data

⚠️ **Note**: Model performance will be lower with sample data due to limited training examples.

**Expected Metrics**:
- Training samples: ~28 (vs 8,448 with full data)
- Validation samples: ~7 (vs 2,112 with full data)
- Training accuracy: ~40-60% (vs 94.2% with full data)
- Validation accuracy: ~30-50% (vs 94.6% with full data)

**Use sample data for**:
- ✅ Testing code changes
- ✅ Validating pipeline
- ✅ Demonstrating workflow

**Use full data for**:
- ✅ Production model training
- ✅ Research and publication
- ✅ Achieving reported accuracy

---

## Regenerating Sample Data

If you have the full dataset and want to regenerate samples:

```bash
# Delete old samples
rm -rf data-sample/input/*/*/*.fff

# Generate new samples (5 files per directory)
python scripts/sample_data.py --fixed-count 5

# Or with percentage sampling
python scripts/sample_data.py --percentage 0.05 --min-files 5 --max-files 10
```

---

## Git Handling

### Included in Repository

✅ `data-sample/input/**/*.fff` - Sample FFF files
✅ `data-sample/README.md` - This file
✅ `data-sample/` directory structure

### Excluded from Repository (Generated)

❌ `data-sample/processed/` - Preprocessed tensors
❌ `data-sample/results/` - Prediction outputs
❌ `data-sample/test/` - Test results

These are regenerated when you run the notebooks.

---

## File Size Comparison

| Component | Sample Size | Full Size | Ratio |
|-----------|-------------|-----------|-------|
| Input FFF files | ~30-50 MB | ~6.5 GB | 0.05% |
| Preprocessed tensors | ~5 MB | ~43 GB | 0.01% |
| Model training time | ~30 sec | ~5-10 min | 5% |
| Total disk space | ~50 MB | ~60 GB | 0.08% |

---

## Limitations

### What Sample Data CAN Do:
✅ Demonstrate code structure
✅ Validate pipeline functionality
✅ Test preprocessing steps
✅ Verify model architecture
✅ Quick iteration during development

### What Sample Data CANNOT Do:
❌ Train production-quality models
❌ Achieve 94.6% validation accuracy
❌ Generalize to real-world conditions
❌ Replace full dataset for research
❌ Provide statistically significant results

---

## Switching Between Sample and Full Data

### Quick Switch Method

**Option 1: Environment Variable** (Recommended)
```python
import os
USE_SAMPLE = os.getenv('USE_SAMPLE_DATA', 'false').lower() == 'true'

data_dir = 'data-sample' if USE_SAMPLE else 'data'
```

Run with:
```bash
USE_SAMPLE_DATA=true jupyter notebook notebooks/01_DataPreprocessing.ipynb
```

**Option 2: Config Profiles**

Create `config/config.sample.yaml`:
```yaml
paths:
  input: "data-sample/input"
  processed: "data-sample/processed"
  results: "data-sample/results"
```

Load with:
```python
from src.config import Config
config = Config('config/config.sample.yaml')
```

**Option 3: Manual Edit**

Edit `config/config.yaml` and change paths between `data/` and `data-sample/`.

---

## Verifying Sample Data

```bash
# Check file counts
find data-sample/input -name "*.fff" | wc -l
# Should show: ~35-40 files

# Check total size
du -sh data-sample/input/
# Should show: ~30-50M

# List files per directory
for dir in data-sample/input/*/*/; do
    echo "$dir: $(ls $dir/*.fff 2>/dev/null | wc -l) files"
done
```

---

## FAQ

**Q: Can I train a production model with sample data?**
A: No. The sample is too small (~35 files). You need the full dataset (10,560 files) for production-quality models.

**Q: Why is my validation accuracy only 40%?**
A: This is expected with sample data. With only ~28 training samples, the model cannot learn effectively. Use full dataset for 94.6% accuracy.

**Q: How do I get the full dataset?**
A: Contact IG-EPN (https://www.igepn.edu.ec/) and request access to the complete thermal imaging dataset.

**Q: Can I add more sample files?**
A: Yes! Run `python scripts/sample_data.py --fixed-count 10` to increase to 10 files per directory. Keep total under 100MB for GitHub.

**Q: Should I commit data-sample/ to git?**
A: Yes! The sample data (~30-50MB) is small enough for git and helps others get started quickly.

---

## Contact

For full dataset access:
**Geophysical Institute - Escuela Politécnica Nacional (IG-EPN)**
Website: https://www.igepn.edu.ec/
Mail: *svallejo@igepn.edu.ec
---

**Last Updated**: February 6, 2025
**Sample Size**: ~35-40 FFF files (~30-50 MB)
**Sampling Rate**: 0.05-0.1% of full dataset
