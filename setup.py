"""Setup configuration for VolcanesML package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="volcanesml",
    version="1.0.0",
    description="Volcanic thermal imaging classification using multi-branch CNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Geophysical Institute - EPN",
    author_email="",
    url="https://github.com/your-org/volcanesml",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "flirpy>=0.3.0",
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "tensorboard": ["tensorboard>=2.8.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="machine-learning pytorch volcano-monitoring thermal-imaging computer-vision",
)
