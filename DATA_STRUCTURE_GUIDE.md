# Data Structure Guide

## ⚠️ CRITICAL: Understanding data/ vs data-sample/

This project uses **two separate data directories** with different purposes.

---

## Quick Reference

| Feature | `data-sample/` | `data/` |
|---------|----------------|---------|
| **In Git?** | ✅ Yes | ❌ No (gitignored) |
| **Size** | ~12-20 MB | ~50-60 GB |
| **Files** | ~35 FFF files | ~10,603 FFF files |
| **Sampling** | 0.33% (5 per class) | 100% (complete) |
| **Use for** | Quick testing, demos | Production training |
| **Accuracy** | ⚠️ ~40-60% | ✅ ~94.6% |
| **Setup** | Already included | Contact IG-EPN |

---

## data-sample/ Directory (INCLUDED)

### Purpose
- ✅ **Included in git repository**
- ✅ **Quick testing and validation**
- ✅ **Code demonstration**
- ✅ **Learning the pipeline**
- ✅ **CI/CD testing**

### Contents
```
data-sample/                   # ~12 MB total
├── input/
│   ├── Cotopaxi/
│   │   ├── Despejado/        # 5 files (from 3,127)
│   │   ├── Emisiones/        # 5 files (from 3,498)
│   │   └── Nublado/          # 5 files (from 401)
│   └── Reventador/
│       ├── Despejado/        # 5 files (from 41)
│       ├── Emisiones/        # 5 files (from 789)
│       ├── Flujo/            # 5 files (from 1,199)
│       └── Nublado/          # 5 files (from 1,548)
├── processed/                 # Generated (gitignored)
├── test/
└── results/                   # Generated (gitignored)
```

### File Count
- **Total**: 35 FFF files
- **Per class**: ~5 files each
- **Sampling rate**: 0.33% of full dataset

### Expected Performance
⚠️ **Low accuracy (40-60%)** due to insufficient training data

**Training metrics**:
- Training samples: ~28
- Validation samples: ~7
- Epochs to convergence: 5-10
- Training time: ~30 seconds

**This is normal and expected!** The sample data is only for testing code, not training production models.

---

## data/ Directory (NOT INCLUDED)

### Purpose
- ❌ **NOT in git repository** (too large)
- ✅ **Production model training**
- ✅ **Research and publication**
- ✅ **Achieving 94.6% accuracy**
- ✅ **Full temporal analysis**

### Contents
```
data/                          # ~50-60 GB total (NOT IN GIT)
├── input/
│   ├── Cotopaxi/
│   │   ├── Despejado/        # 3,127 files
│   │   ├── Emisiones/        # 3,498 files
│   │   └── Nublado/          # 401 files
│   └── Reventador/
│       ├── Despejado/        # 41 files
│       ├── Emisiones/        # 789 files
│       ├── Flujo/            # 1,199 files
│       └── Nublado/          # 1,548 files
├── processed/                 # ~43 GB (generated)
│   ├── train/
│   ├── val/
│   └── models/
└── results/
```

### File Count
- **Total**: 10,603 FFF files
- **Cotopaxi**: 7,026 files
- **Reventador**: 3,577 files

### Expected Performance
✅ **High accuracy (94.6%)** with full dataset

**Training metrics**:
- Training samples: ~8,448
- Validation samples: ~2,112
- Epochs to convergence: 20-25
- Training time: ~5-10 minutes

### Obtaining Full Dataset
**Contact**: Geophysical Institute - Escuela Politécnica Nacional (IG-EPN)
**Website**: https://www.igepn.edu.ec/
**Email**: Contact via website
**Purpose**: Available for research and educational use

---

## Configuration

### Using data-sample/ (Default for new clones)

If you just cloned the repository, update `config/config.yaml`:

```yaml
paths:
  input: "data-sample/input"
  processed: "data-sample/processed"
  results: "data-sample/results"
```

### Using data/ (Production)

If you have the full dataset, use:

```yaml
paths:
  input: "data/input"
  processed: "data/processed"
  results: "data/results"
```

---

## Switching Between Datasets

### Method 1: Edit config.yaml (Recommended)

Edit `config/config.yaml` and change the paths:

```yaml
# For sample data
paths:
  input: "data-sample/input"

# For full data
paths:
  input: "data/input"
```

### Method 2: Environment Variable

```python
import os
USE_SAMPLE = os.getenv('USE_SAMPLE_DATA', 'false').lower() == 'true'
data_dir = 'data-sample' if USE_SAMPLE else 'data'
```

Run with:
```bash
USE_SAMPLE_DATA=true jupyter notebook notebooks/01_DataPreprocessing.ipynb
```

### Method 3: Command Line Argument

```bash
python train.py --data-dir data-sample/
python train.py --data-dir data/
```

---

## Git Handling

### .gitignore Rules

```gitignore
# Ignore full dataset (too large)
data/

# Keep sample data
!data-sample/
```

### What Gets Committed

✅ **Included in git**:
- `data-sample/input/**/*.fff` (~12 MB)
- `data-sample/README.md`
- `data-sample/` directory structure

❌ **Excluded from git**:
- `data/` entire directory (~60 GB)
- `data-sample/processed/` (generated files)
- `data-sample/results/` (generated files)

---

## Regenerating Sample Data

If you have the full dataset and want to regenerate samples:

### Basic Usage
```bash
# 5 files per directory (default)
python scripts/sample_data.py --fixed-count 5

# 10 files per directory
python scripts/sample_data.py --fixed-count 10

# Percentage-based sampling
python scripts/sample_data.py --percentage 0.1 --min-files 5
```

### Advanced Options
```bash
# Custom minimum/maximum
python scripts/sample_data.py --fixed-count 10 --max-files 15

# Percentage with constraints
python scripts/sample_data.py --percentage 0.05 --min-files 3 --max-files 10
```

### Script Output
```
======================================================================
FFF File Sampling for data-sample Directory
======================================================================
Mode: Fixed count (5 files per directory)

✓ Cotopaxi/Despejado:
    Original: 3,127 files
    Sampled: 5 files (0.16%)
    Files: CTP_RUMHIR_20220805_1950.fff, ...

...

======================================================================
Summary:
  Total original files: 10,603
  Total sampled files: 35
  Overall sampling rate: 0.330%
  Estimated sample size: ~21.0 MB
======================================================================
```

---

## Verification

### Check Sample Data
```bash
# File count
find data-sample/input -name "*.fff" | wc -l
# Should show: 35

# Total size
du -sh data-sample/
# Should show: ~12-20M

# Files per directory
for dir in data-sample/input/*/*/; do
    echo "$dir: $(ls $dir/*.fff 2>/dev/null | wc -l) files"
done
# Each should show: 5 files
```

### Check Full Data (if available)
```bash
# File count
find data/input -name "*.fff" | wc -l
# Should show: ~10,603

# Total size
du -sh data/
# Should show: ~50-60G

# Class distribution
for dir in data/input/*/*/; do
    echo "$dir: $(ls $dir/*.fff 2>/dev/null | wc -l) files"
done
```

---

## Common Issues

### Issue: "No files found in data/"
**Solution**: You need to obtain the full dataset from IG-EPN or use `data-sample/` for testing.

### Issue: "Low accuracy (~40%) during training"
**Solution**: This is expected with `data-sample/`. Use full `data/` for production training (94.6% accuracy).

### Issue: "Config still points to data/"
**Solution**: Edit `config/config.yaml` and change paths to `data-sample/`.

### Issue: "Git trying to commit large files"
**Solution**: Verify `.gitignore` is present and contains:
```gitignore
data/
!data-sample/
```

### Issue: "Cannot regenerate samples"
**Solution**: You need the full `data/` directory. Contact IG-EPN to obtain it.

---

## Best Practices

### For Development
✅ Use `data-sample/` for:
- Testing code changes
- Validating pipeline
- Quick iteration
- CI/CD testing

### For Production
✅ Use `data/` for:
- Training production models
- Research and publication
- Achieving 94.6% accuracy
- Full dataset analysis

### For Version Control
✅ **Commit**: `data-sample/` (small, ~12MB)
❌ **Don't commit**: `data/` (large, ~60GB)

### For Sharing
- Share code + `data-sample/` via GitHub
- Share full `data/` via direct transfer or cloud storage
- Document how to obtain full dataset in README

---

## Performance Comparison

| Metric | data-sample/ | data/ |
|--------|--------------|-------|
| **Training samples** | ~28 | ~8,448 |
| **Validation samples** | ~7 | ~2,112 |
| **Training time** | ~30 sec | ~5-10 min |
| **Train accuracy** | ~50-70% | ~94.2% |
| **Val accuracy** | ⚠️ ~40-60% | ✅ ~94.6% |
| **Overfitting risk** | High | Low |
| **Generalization** | Poor | Excellent |
| **Production ready** | ❌ No | ✅ Yes |

---

## FAQs

**Q: Should I use data-sample/ or data/ for training?**
A: Use `data-sample/` for testing code, `data/` for production models.

**Q: Why is accuracy only 40% with data-sample/?**
A: 35 images is insufficient for deep learning. This is expected and normal.

**Q: Can I add more files to data-sample/?**
A: Yes, run `python scripts/sample_data.py --fixed-count 10` for 10 files per class (~24MB).

**Q: How do I get the full data/?**
A: Contact IG-EPN (https://www.igepn.edu.ec/) and request access.

**Q: Will data/ be committed to git?**
A: No, it's in `.gitignore`. Only `data-sample/` is committed.

**Q: Can I use data-sample/ for production?**
A: ❌ No. You need full `data/` for 94.6% accuracy.

---

## Summary

- ✅ **data-sample/** = Small sample for testing (INCLUDED in git)
- ❌ **data/** = Full dataset for production (NOT in git)
- 📝 Use **config.yaml** to switch between them
- 🎯 Expect low accuracy with sample data (this is normal)
- 📧 Contact **IG-EPN** to obtain full dataset

---

**Created**: February 6, 2025
**Sample Files**: 35 FFF files (~12 MB)
**Full Dataset**: 10,603 FFF files (~50-60 GB)
**Sampling Rate**: 0.33%
