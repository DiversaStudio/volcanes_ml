"""Setup verification script for VolcanesML.

This script checks that the development environment is correctly configured
and all dependencies are installed properly.

Usage:
    python scripts/verify_setup.py
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_python_version():
    """Check Python version >= 3.8"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_dependencies():
    """Check that all required packages are installed."""
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'scikit-learn'),
        ('flirpy', 'flirpy'),
        ('tqdm', 'tqdm'),
        ('yaml', 'PyYAML'),
    ]

    all_installed = True
    print("\nDependency Check:")
    print("-" * 50)

    for package_name, display_name in required_packages:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name:20s} ({version})")
        except ImportError:
            print(f"✗ {display_name:20s} (NOT INSTALLED)")
            all_installed = False

    return all_installed

def check_cuda_availability():
    """Check CUDA availability (informational, not required)."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n✓ CUDA available: {cuda_version}")
            print(f"  GPU: {gpu_name}")
            return True
        else:
            print("\n⚠ CUDA not available (CPU-only mode)")
            print("  Training will be slower but functional")
            return False
    except ImportError:
        print("\n✗ Cannot check CUDA (PyTorch not installed)")
        return False

def check_directory_structure():
    """Verify project directory structure exists."""
    required_dirs = [
        'config',
        'data/input',
        'data/processed',
        'data/test',
        'data/results',
        'figures',
        'notebooks',
        'scripts',
        'src',
        'src/data',
        'src/features',
        'src/models',
    ]

    all_exist = True
    print("\nDirectory Structure:")
    print("-" * 50)

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (MISSING)")
            all_exist = False

    return all_exist

def check_config_file():
    """Check that config.yaml exists and is valid."""
    config_path = project_root / 'config' / 'config.yaml'

    print("\nConfiguration File:")
    print("-" * 50)

    if not config_path.exists():
        print(f"✗ config/config.yaml not found")
        return False

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check essential keys
        required_keys = ['paths', 'data', 'model', 'training', 'labels']
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            print(f"✗ config.yaml missing keys: {', '.join(missing_keys)}")
            return False

        print(f"✓ config/config.yaml is valid")

        # Check label mapping has 4 classes
        if len(config.get('labels', {})) != 4:
            print(f"⚠ config.yaml has {len(config['labels'])} classes (expected 4)")
            return False

        return True

    except Exception as e:
        print(f"✗ Error reading config.yaml: {e}")
        return False

def check_sample_data():
    """Check that sample data exists."""
    input_dir = project_root / 'data' / 'input'

    print("\nSample Data:")
    print("-" * 50)

    if not input_dir.exists():
        print(f"✗ data/input directory not found")
        return False

    # Count FFF files
    fff_files = list(input_dir.rglob('*.fff'))
    num_files = len(fff_files)

    if num_files == 0:
        print(f"⚠ No sample FFF files found in data/input/")
        print(f"  Download sample data or run with your own FFF files")
        return False
    elif num_files < 10:
        print(f"⚠ Only {num_files} sample FFF files found")
        print(f"  Consider adding more samples for better testing")
        return True
    else:
        print(f"✓ Found {num_files} sample FFF files")
        return True

def check_source_files():
    """Check that essential source files exist."""
    required_files = [
        'src/config.py',
        'src/data/loader.py',
        'src/data/preprocessor.py',
        'src/data/dataset.py',
        'src/features/edge_detection.py',
        'src/features/thermal.py',
        'src/models/multibranch.py',
        'src/models/trainer.py',
        'src/models/predictions.py',
    ]

    all_exist = True
    print("\nSource Files:")
    print("-" * 50)

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (MISSING)")
            all_exist = False

    return all_exist

def test_config_loading():
    """Test that configuration can be loaded."""
    print("\nConfiguration Loading:")
    print("-" * 50)

    try:
        from src.config import get_config
        config = get_config()

        print(f"✓ Configuration loaded successfully")
        print(f"  Project root: {config.project_root}")
        print(f"  Model classes: {config.model['n_classes']}")
        print(f"  Training epochs: {config.training['n_epochs']}")

        return True

    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("VolcanesML Setup Verification")
    print("=" * 60)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Configuration File", check_config_file),
        ("Source Files", check_source_files),
        ("Configuration Loading", test_config_loading),
        ("Sample Data", check_sample_data),
    ]

    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n✗ {check_name} check failed with error: {e}")
            results[check_name] = False

    # CUDA is informational only
    check_cuda_availability()

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {check_name}")

    print("-" * 60)
    print(f"Passed: {passed}/{total} checks")

    if passed == total:
        print("\n✓ All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("  1. Run notebooks in order: 00 → 01 → 02 → 03 → 04")
        print("  2. Start with 00_VolcanoData.ipynb for data exploration")
        print("  3. See README.md for detailed usage instructions")
        return 0
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check directory structure matches documentation")
        print("  - Ensure config/config.yaml is properly configured")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
