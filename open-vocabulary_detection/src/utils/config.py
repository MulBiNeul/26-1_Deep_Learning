import yaml

def load_config(path: str) -> dict:
    """
    Read YAML settings and convert to dict

    Args:
        path (str): YAML file path
    
    Returns:
        dict: setting value dict
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config