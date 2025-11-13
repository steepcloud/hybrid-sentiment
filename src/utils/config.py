import yaml
from typing import Dict, Any, Optional
import os
from pathlib import Path


class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.config = self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Configuration loaded from {config_path}")
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports nested keys with dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports nested keys with dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, save_path: Optional[str] = None):
        """
        Save configuration to YAML file.
        
        Args:
            save_path: Optional path to save (defaults to original path)
        """
        if save_path is None:
            save_path = self.config_path
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to {save_path}")
    
    def print_config(self):
        """Print current configuration."""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        self._print_dict(self.config)
        print("="*60)
    
    def _print_dict(self, d: Dict, indent: int = 0):
        """Helper to print nested dictionary."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    def validate_config(self) -> bool:
        """
        Validate configuration has required fields.
        
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'project.random_seed',
            'data.vocab_size',
            'data.max_length',
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                print(f"Missing required field: {field}")
                return False
        
        return True
    
    def update_from_dict(self, updates: Dict):
        """
        Update configuration from dictionary.
        
        Args:
            updates: Dictionary with updates
        """
        def recursive_update(config, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    recursive_update(config[key], value)
                else:
                    config[key] = value
        
        recursive_update(self.config, updates)
        print("Configuration updated")


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_value(config: Dict, key: str, default: Any = None) -> Any:
    """
    Get value from nested config dictionary.
    
    Args:
        config: Configuration dictionary
        key: Key with dot notation (e.g., 'data.vocab_size')
        default: Default value if not found
        
    Returns:
        Configuration value
    """
    keys = key.split('.')
    value = config
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


if __name__ == "__main__":
    # Test configuration manager
    print("Testing Configuration Manager...")
    
    # Load config
    print("\n" + "="*60)
    config = Config("configs/config.yaml")
    
    # Print config
    config.print_config()
    
    # Test get
    print("\n" + "="*60)
    print("Testing get method:")
    print(f"Random seed: {config.get('project.random_seed')}")
    print(f"Vocab size: {config.get('data.vocab_size')}")
    print(f"Non-existent key: {config.get('nonexistent.key', 'default_value')}")
    
    # Test set
    print("\n" + "="*60)
    print("Testing set method:")
    config.set('test.new_value', 42)
    print(f"New value: {config.get('test.new_value')}")
    
    # Test validation
    print("\n" + "="*60)
    print(f"Config valid: {config.validate_config()}")
    
    # Test update from dict
    print("\n" + "="*60)
    updates = {
        'project': {
            'random_seed': 123
        },
        'new_section': {
            'value': 'test'
        }
    }
    config.update_from_dict(updates)
    print(f"Updated random seed: {config.get('project.random_seed')}")
    print(f"New section value: {config.get('new_section.value')}")
    
    # Test save
    print("\n" + "="*60)
    os.makedirs('results/test', exist_ok=True)
    config.save_config('results/test/test_config.yaml')
    
    print("\n✓ All tests passed!")