#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample FFF files from full dataset to create data-sample directory.

This script samples a small percentage of files from each class to create
a lightweight sample dataset suitable for GitHub and quick testing.

Usage:
    python scripts/sample_data.py --percentage 0.05 --min-files 5
    python scripts/sample_data.py --fixed-count 5
"""
import os
import sys
import shutil
import random
import argparse
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def get_fff_files(directory):
    """Get all .fff files in a directory."""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith('.fff')]


def sample_files(files, percentage=None, fixed_count=None, min_files=3):
    """
    Sample files from a list.

    Args:
        files: List of filenames
        percentage: Percentage to sample (e.g., 0.05 for 0.05%)
        fixed_count: Fixed number of files to sample
        min_files: Minimum number of files to sample

    Returns:
        List of sampled filenames
    """
    if not files:
        return []

    if fixed_count:
        count = min(fixed_count, len(files))
    elif percentage:
        count = max(min_files, int(len(files) * percentage / 100))
        count = min(count, len(files))
    else:
        count = min(min_files, len(files))

    # Sort files for reproducibility, then sample
    sorted_files = sorted(files)
    random.seed(42)  # Fixed seed for reproducibility

    # Sample evenly across the sorted list for better representation
    step = len(sorted_files) // count if count > 0 else 1
    sampled = [sorted_files[i * step] for i in range(count)]

    return sampled


def copy_files(source_dir, dest_dir, files):
    """Copy files from source to destination."""
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0

    for filename in files:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        try:
            shutil.copy2(source_path, dest_path)
            copied += 1
        except Exception as e:
            print(f"  Warning: Could not copy {filename}: {e}")

    return copied


def main():
    parser = argparse.ArgumentParser(
        description='Sample FFF files from full dataset to create data-sample directory'
    )
    parser.add_argument(
        '--percentage',
        type=float,
        default=None,
        help='Percentage of files to sample (e.g., 0.05 for 0.05%%)'
    )
    parser.add_argument(
        '--fixed-count',
        type=int,
        default=None,
        help='Fixed number of files to sample per directory'
    )
    parser.add_argument(
        '--min-files',
        type=int,
        default=5,
        help='Minimum number of files to sample per directory (default: 5)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to sample per directory'
    )

    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Define directories to sample
    directories = [
        ('Cotopaxi', 'Despejado'),
        ('Cotopaxi', 'Emisiones'),
        ('Cotopaxi', 'Nublado'),
        ('Reventador', 'Despejado'),
        ('Reventador', 'Emisiones'),
        ('Reventador', 'Flujo'),
        ('Reventador', 'Nublado'),
    ]

    print("=" * 70)
    print("FFF File Sampling for data-sample Directory")
    print("=" * 70)

    if args.fixed_count:
        print(f"Mode: Fixed count ({args.fixed_count} files per directory)")
    elif args.percentage:
        print(f"Mode: Percentage sampling ({args.percentage}%)")
        print(f"Minimum files per directory: {args.min_files}")
    else:
        print(f"Mode: Default (minimum {args.min_files} files per directory)")

    print()

    total_original = 0
    total_sampled = 0

    for volcano, label in directories:
        source_dir = project_root / 'data' / 'input' / volcano / label
        dest_dir = project_root / 'data-sample' / 'input' / volcano / label

        # Get files
        files = get_fff_files(source_dir)
        total_original += len(files)

        if not files:
            print(f"⚠️  {volcano}/{label}: No files found (skipping)")
            continue

        # Sample files
        sampled_files = sample_files(
            files,
            percentage=args.percentage,
            fixed_count=args.fixed_count,
            min_files=args.min_files
        )

        if args.max_files:
            sampled_files = sampled_files[:args.max_files]

        # Copy files
        copied = copy_files(source_dir, dest_dir, sampled_files)
        total_sampled += copied

        percentage_sampled = (copied / len(files) * 100) if files else 0

        print(f"✓ {volcano}/{label}:")
        print(f"    Original: {len(files):,} files")
        print(f"    Sampled: {copied} files ({percentage_sampled:.2f}%)")
        print(f"    Files: {', '.join(sampled_files[:3])}" +
              (f", ..." if len(sampled_files) > 3 else ""))
        print()

    print("=" * 70)
    print("Summary:")
    print(f"  Total original files: {total_original:,}")
    print(f"  Total sampled files: {total_sampled}")
    print(f"  Overall sampling rate: {(total_sampled/total_original*100):.3f}%")

    # Estimate size
    avg_file_size = 0.6  # MB per file
    estimated_size = total_sampled * avg_file_size
    print(f"  Estimated sample size: ~{estimated_size:.1f} MB")
    print("=" * 70)
    print()
    print(f"✓ Sample data created in: {project_root / 'data-sample'}")
    print()
    print("Next steps:")
    print("  1. Verify sample data: ls -lh data-sample/input/*/*/")
    print("  2. Test with notebooks using config: limit_per_directory: null")
    print("  3. Commit to git: git add data-sample/")


if __name__ == '__main__':
    main()
