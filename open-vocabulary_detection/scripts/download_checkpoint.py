import os
import sys
import yaml

# Add project root to import path if scripts/download_checkpoint.py is directly executed
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

########################################################################################

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.utils.config import load_config

def main():
    """
    Save model and processor beforehand to the local cache

    Execution example:
        python scripts/download_checkpoint.py --config configs/default.yaml
    """
    import argparse

    parser = argparse.ArgumentParser(
        description = "Downloading MM Grounding DINO checkpoint and processor"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = cfg["model"]["model_id"]

    print(f"Downloading processor: {model_id}")
    AutoProcessor.from_pretrained(model_id)

    print(f"Downloading model: {model_id}")
    AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    print("Download complete!")

if __name__ == "__main__":
    main()