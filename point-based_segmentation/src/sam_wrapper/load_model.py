from pathlib import Path
from transformers import SamModel, SamProcessor

from src.utils.device import get_device


def load_sam_model(local_model_dir: str, requested_device: str = "auto"):
    """
    Load SAM model and processor from a local directory.

    Args:
        local_model_dir (str): Path to the saved SAM model directory
        requested_device (str): Device preference ('auto', 'cuda', 'mps', 'cpu')

    Returns:
        processor: SAM processor
        model: SAM model
        device: selected device
    """

    model_dir = Path(local_model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Please run `python scripts/download_checkpoint.py` first."
        )

    device = get_device(requested_device)

    processor = SamProcessor.from_pretrained(str(model_dir))
    model = SamModel.from_pretrained(str(model_dir))

    model.to(device)
    model.eval()

    return processor, model, device