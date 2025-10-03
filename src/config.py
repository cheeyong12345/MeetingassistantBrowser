import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._ensure_directories()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.storage.data_dir,
            self.storage.meetings_dir,
            self.storage.models_dir
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self):
        """Save current configuration to file"""
        with open(self.config_path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False)

    @property
    def app(self):
        return ConfigSection(self._config.get('app', {}))

    @property
    def server(self):
        return ConfigSection(self._config.get('server', {}))

    @property
    def audio(self):
        return ConfigSection(self._config.get('audio', {}))

    @property
    def stt(self):
        return ConfigSection(self._config.get('stt', {}))

    @property
    def summarization(self):
        return ConfigSection(self._config.get('summarization', {}))

    @property
    def storage(self):
        return ConfigSection(self._config.get('storage', {}))

    @property
    def processing(self):
        return ConfigSection(self._config.get('processing', {}))

class ConfigSection:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str):
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        raise AttributeError(f"'ConfigSection' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._data.copy()

# Global configuration instance
config = Config()