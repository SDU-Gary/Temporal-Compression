"""Configuration utilities for loading and validating YAML configs."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


class Config:
    """Configuration container for training parameters.

    Loads configuration from YAML file and provides easy access.
    """

    def __init__(self, config_dict: Dict):
        """Initialize from configuration dictionary.

        Args:
            config_dict: Dictionary loaded from YAML file
        """
        self.config = config_dict

        # Store top-level sections for easy access
        self.experiment = config_dict.get('experiment', {})
        self.data = config_dict.get('data', {})
        self.model = config_dict.get('model', {})
        self.training = config_dict.get('training', {})
        self.validation = config_dict.get('validation', {})
        self.gaussian_init = config_dict.get('gaussian_init', {})

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> 'Config':
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            config: Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    def save(self, output_path: Path | str):
        """Save configuration to YAML file.

        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation.

        Examples:
            config.get('model.num_gaussians')
            config.get('training.optimizer.gaussians.lr')

        Args:
            key: Dot-separated key path
            default: Default value if key not found

        Returns:
            value: Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return self.config[key]

    def __repr__(self) -> str:
        return f"Config({list(self.config.keys())})"


def validate_config(config: Config) -> bool:
    """Validate configuration has all required fields.

    Args:
        config: Configuration to validate

    Returns:
        valid: True if configuration is valid

    Raises:
        ValueError: If configuration is missing required fields
    """
    required_fields = [
        'experiment.name',
        'experiment.output_dir',
        'experiment.device',
        'data.dataset_path',
        'data.batch_size',
        'model.num_gaussians',
        'model.latent_dim_base',
        'model.latent_dim_time',
        'training.num_steps',
    ]

    missing_fields = []
    for field in required_fields:
        if config.get(field) is None:
            missing_fields.append(field)

    if missing_fields:
        raise ValueError(f"Missing required configuration fields: {missing_fields}")

    return True


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Recursively merge two configuration dictionaries.

    Override config takes precedence over base config.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        merged: Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def validate_with_schema(config: Dict[str, Any], schema_path: Path | str, strict: bool = False) -> list[str]:
    """Validate a config dict against a JSON schema if jsonschema is available.

    Returns a list of warning/error messages. If strict=True, raises ValueError on violations.
    """
    schema_path = Path(schema_path)
    messages: list[str] = []
    try:
        import jsonschema
    except Exception:
        messages.append("jsonschema not installed; skipping schema validation")
        return messages

    if not schema_path.exists():
        messages.append(f"schema not found: {schema_path}")
        return messages

    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)
    except Exception as exc:
        messages.append(f"failed to load schema: {exc}")
        return messages

    try:
        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(config), key=lambda e: e.path)
        for err in errors:
            path = ".".join([str(p) for p in err.path])
            prefix = f"{path}: " if path else ""
            messages.append(prefix + err.message)
        if errors and strict:
            raise ValueError("schema validation failed")
    except Exception as exc:
        if strict:
            raise
        messages.append(f"schema validation error: {exc}")
    return messages
