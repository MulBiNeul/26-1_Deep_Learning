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

def save_image(image, svae_dir: str, filename: str) -> str:
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
    os.makedirs(svae_dir, exist_ok=True)
    save_path = os.path.join(save_path, filename)
    image.save(save_path)
    print(f"saved image: {save_path}")

    return save_path