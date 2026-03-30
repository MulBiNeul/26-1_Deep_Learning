from pathlib import Path
from PIL import Image


def load_image(image_path: str) -> Image.Image:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"입력 이미지를 찾을 수 없습니다: {image_path}")

    return Image.open(image_path).convert("RGB")

def resize_image(image, target_size: int):
    """
    긴 변을 기준으로 resize (aspect ratio 유지)
    """
    w, h = image.size

    if max(w, h) == target_size:
        return image, 1.0

    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = image.resize((new_w, new_h))

    return resized, scale

def ensure_dir(dir_path: str):
    Path(dir_path).mkdir(parents=True, exist_ok=True)