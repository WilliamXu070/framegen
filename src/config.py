"""
Configuration management for the Frame Generation Application.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path_or_dict):
        """
        Initialize configuration from YAML file or dictionary.
        
        Args:
            config_path_or_dict: Path to the configuration YAML file or dictionary
        """
        self.config_path = config_path_or_dict if isinstance(config_path_or_dict, str) else None
        self.config = self._load_config(config_path_or_dict)
        self._create_directories()
    
    def _load_config(self, config_path_or_dict) -> Dict[str, Any]:
        """Load configuration from YAML file or dictionary."""
        if isinstance(config_path_or_dict, dict):
            return config_path_or_dict
        elif isinstance(config_path_or_dict, str):
            try:
                with open(config_path_or_dict, 'r') as file:
                    config = yaml.safe_load(file)
                return config
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file not found: {config_path_or_dict}")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing configuration file: {e}")
        else:
            raise ValueError("config_path_or_dict must be a string (file path) or dictionary")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        paths = self.config.get('paths', {})
        for key, path in paths.items():
            if key.endswith('_dir'):
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation like 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save configuration (defaults to original path)
        """
        save_path = path or self.config_path
        with open(save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('model', {})
    
    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    @property
    def paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.get('paths', {})
