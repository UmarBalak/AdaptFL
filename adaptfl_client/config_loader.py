import json
import os

class ConfigLoader:
    """
    Loads and validates the configuration for the client framework.
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        with open(self.config_path, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing configuration file: {e}")

        # Perform basic validation
        required_keys = ["client_id", "dataset_path", "local_model_dir", "azure_connection_string", "server_url"]
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")
        
        return config

    def get(self, key, default=None):
        """
        Retrieve a value from the config.
        """
        return self.config.get(key, default)
