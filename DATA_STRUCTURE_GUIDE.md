# Data Structure Guide

## ⚠️ CRITICAL: Understanding data/ vs data-sample/

This project uses **two separate data directories** with different purposes.

---

## Quick Reference

| Feature | `data-sample/` | `data/` |
|---------|----------------|---------|
| **In Git?** | ✅ Yes | ❌ No (gitignored) |
| **Size** | ~12-20 MB | ~15 GB |
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
**Contact**: Institito Geofísico-EPN
**Website**: https://www.igepn.edu.ec/
**Email**: *svallejo@igepn.edu.ec
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
- `data/` entire directory (~15 GB)
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
