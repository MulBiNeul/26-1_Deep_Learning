import yaml


def load_config(config_path: str) -> dict: 
    """
    Load a YAML config file and return it as a dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)