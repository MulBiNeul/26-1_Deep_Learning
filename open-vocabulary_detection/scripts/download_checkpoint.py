import os
import sys
import argparse

# Add project root to sys.path when this script is executed directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download OWLv2 / OWL-ViT checkpoint and processor"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main():
    """
    Download and cache the selected model checkpoint and processor in advance.

    Example:
        python scripts/download_checkpoint.py --config configs/default.yaml
    """
    args = parse_args()
    cfg = load_config(args.config)

    model_name = cfg["model"]["model_name"].lower()
    model_id = cfg["model"]["model_id"]

    print(f"Downloading processor: {model_id}")

    if model_name == "owlvit":
        OwlViTProcessor.from_pretrained(model_id)
        print(f"Downloading model: {model_id}")
        OwlViTForObjectDetection.from_pretrained(model_id)

    elif model_name == "owlv2":
        Owlv2Processor.from_pretrained(model_id)
        print(f"Downloading model: {model_id}")
        Owlv2ForObjectDetection.from_pretrained(model_id)

    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Use 'owlvit' or 'owlv2'."
        )

    print("Download complete!")
    print("The files are now cached locally in the Hugging Face cache directory.")


if __name__ == "__main__":
    main()