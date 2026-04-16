import os
from PIL import Image

def load_image(image_path: str) -> Image.Image:
    """
    Load image and convert to RGB format

    Args:
        image_path (str): input image path
    
    Returns:
        PIL.Image.Image: RGB image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return Image.open(image_path).convert("RGB")

def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Resize image while preserving aspect ratio.

    Args:
        image (PIL.Image): input image
        max_size (int): maximum size for longer side

    Returns:
        resized PIL.Image
    """
    width, height = image.size

    # If already small enough, return original
    if max(width, height) <= max_size:
        return image

    scale = max_size / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = image.resize((new_width, new_height), Image.BILINEAR)

    print(f"Image resized: ({width}, {height}) -> ({new_width}, {new_height})")

    return resized


def save_image(image: Image.Image, save_dir: str, filename: str) -> str:
    """
    Save PIL image into designated folder

    Args:
        image:
            PIL image to save
        
        save_dir (str):
            folder path to save
        
        filename (str):
            filename to save

    Returns:
        str: final save path
    """
    # if folder does not exist
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    image.save(save_path)
    print(f"saved image: {save_path}")

    return save_path