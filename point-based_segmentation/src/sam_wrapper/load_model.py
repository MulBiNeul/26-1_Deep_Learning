from pathlib import Path
from transformers import SamModel, SamProcessor

from src.utils.device import get_device


def load_sam_model(local_model_dir: str, requested_device: str = "auto"):
    model_dir = Path(local_model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"모델 폴더를 찾을 수 없습니다: {model_dir}\n"
            "먼저 `python scripts/download_checkpoint.py`를 실행하세요."
        )

    device = get_device(requested_device)

    processor = SamProcessor.from_pretrained(str(model_dir))
    model = SamModel.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    return processor, model, device