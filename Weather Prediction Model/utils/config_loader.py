import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    def __init__(self, config_path: str = "config_future.yaml"):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_data_config(self):
        return self.config.get('data', {})

    def get_cities_config(self):
        return self.config.get('cities', {})

    def get_thresholds_config(self):
        return self.config.get('thresholds', {})

    def get_model_config(self, name):
        return self.config.get('models', {}).get(name, {})

    def get_training_config(self):
        return self.config.get('training', {})


def load_config(path="config.yaml"):
    return ConfigLoader(path)