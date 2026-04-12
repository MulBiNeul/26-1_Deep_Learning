from pathlib import Path
from PIL import Image


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from the given path and convert it to RGB format.

    Args:
        image_path (str): Path to the input image

    Returns:
        Image.Image: Loaded RGB image
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def resize_image(image: Image.Image, target_size: int):
    """
    Resize the image while preserving aspect ratio,
    based on the longer side.

    Args:
        image (Image.Image): Input image
        target_size (int): Target size for the longer side

    Returns:
        Tuple[Image.Image, float]: Resized image and scaling factor
    """

    width, height = image.size
    long_side = max(width, height)

    if long_side == target_size:
        return image, 1.0

    scale = target_size / long_side
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = image.resize((new_width, new_height))

    return resized, scale


def ensure_dir(dir_path: str):
    """
    Create a directory if it does not exist.

    Args:
        dir_path (str): Directory path to create
    """

    Path(dir_path).mkdir(parents=True, exist_ok=True)