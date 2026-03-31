import yaml
from pathlib import Path

class ConfigLoader:
    """Load and validate YAML cofiguration files"""
    def __init__(self, config_path, str):
        """
        Initialize the config loader.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)

    def load(self) -> dict:
        """
        Load the YAML configuration file

        Returns:
            dict: Parsed configuration dictionary
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the YAML file is empty.
        """
        if not self.config_path_is_exist():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with self.config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        
        if config is None:
            raise ValueError(f"Configuration file is empty: {self.config_path}")
        
        return config
    