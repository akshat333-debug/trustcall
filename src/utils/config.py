import yaml
import os

def load_config(config_path):
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_configs(base_config, override_config):
    """Recursively merge dictionary configurations."""
    for k, v in override_config.items():
        if isinstance(v, dict) and k in base_config:
            merge_configs(base_config[k], v)
        else:
            base_config[k] = v
    return base_config
