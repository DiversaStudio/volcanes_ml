"""Configuration management for VolcanesML.

This module provides centralized access to project configuration stored in config/config.yaml.
It eliminates hardcoded paths and parameters throughout the codebase.

Usage:
    from src.config import get_config

    config = get_config()
    data_path = config.paths['input']
    batch_size = config.training['batch_size']
"""
import os
from pathlib import Path
import yaml


class Config:
    """Configuration loader and accessor for VolcanesML.

    Attributes:
        config_path (Path): Path to the configuration YAML file
        _config (dict): Loaded configuration dictionary
    """

    def __init__(self, config_path=None):
        """Initialize configuration loader.

        Args:
            config_path (str or Path, optional): Path to config.yaml.
                If None, uses default location at project_root/config/config.yaml
        """
        if config_path is None:
            # Auto-detect project root (parent of src/)
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at: {self.config_path}\n"
                f"Please ensure config/config.yaml exists in the project root."
            )

        self._config = self._load_config()
        self._project_root = Path(__file__).parent.parent

    def _load_config(self):
        """Load configuration from YAML file.

        Returns:
            dict: Configuration dictionary
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        """Get configuration value using dot notation.

        Args:
            key (str): Configuration key in dot notation (e.g., 'paths.data_root')
            default: Default value if key not found

        Returns:
            Value from configuration or default

        Examples:
            >>> config = Config()
            >>> config.get('training.batch_size')
            250
            >>> config.get('paths.input')
            'data/input'
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def get_absolute_path(self, relative_path):
        """Convert relative path from config to absolute path.

        Args:
            relative_path (str): Relative path from config (e.g., 'data/input')

        Returns:
            Path: Absolute path

        Examples:
            >>> config = Config()
            >>> config.get_absolute_path('data/input')
            Path('/full/path/to/project/data/input')
        """
        return self._project_root / relative_path

    @property
    def paths(self):
        """Get paths configuration section.

        Returns:
            dict: Paths configuration
        """
        return self._config.get('paths', {})

    @property
    def model(self):
        """Get model configuration section.

        Returns:
            dict: Model architecture configuration
        """
        return self._config.get('model', {})

    @property
    def training(self):
        """Get training configuration section.

        Returns:
            dict: Training hyperparameters
        """
        return self._config.get('training', {})

    @property
    def data(self):
        """Get data configuration section.

        Returns:
            dict: Data loading settings
        """
        return self._config.get('data', {})

    @property
    def image(self):
        """Get image configuration section.

        Returns:
            dict: Image dimensions
        """
        return self._config.get('image', {})

    @property
    def thresholds(self):
        """Get thermal thresholds configuration.

        Returns:
            dict: Volcano-specific thermal thresholds
        """
        return self._config.get('thresholds', {})

    @property
    def labels(self):
        """Get label mapping configuration.

        Returns:
            dict: Label name to integer mapping
        """
        return self._config.get('labels', {})

    @property
    def device(self):
        """Get device configuration.

        Returns:
            str: Device setting ('auto', 'cuda', or 'cpu')
        """
        return self._config.get('device', 'auto')

    @property
    def project_root(self):
        """Get project root directory.

        Returns:
            Path: Absolute path to project root
        """
        return self._project_root

    def __repr__(self):
        """String representation of Config object."""
        return f"Config(config_path='{self.config_path}')"


# Global configuration instance
_config = None


def get_config(config_path=None):
    """Get or create global configuration instance.

    This function implements a singleton pattern to ensure only one
    configuration is loaded per session.

    Args:
        config_path (str or Path, optional): Path to config.yaml.
            Only used on first call.

    Returns:
        Config: Global configuration instance

    Examples:
        >>> from src.config import get_config
        >>> config = get_config()
        >>> batch_size = config.training['batch_size']
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reload_config(config_path=None):
    """Force reload of configuration.

    Useful for testing or when config file has been modified.

    Args:
        config_path (str or Path, optional): Path to config.yaml

    Returns:
        Config: Reloaded configuration instance
    """
    global _config
    _config = Config(config_path)
    return _config
