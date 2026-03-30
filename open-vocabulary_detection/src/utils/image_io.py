import os

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