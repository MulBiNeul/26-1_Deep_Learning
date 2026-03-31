from pathlib import Path
from PIL import Image

class ImageLoader:
    """ Load images for vision-language inference """

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """
        Load an image and convert it to RGB format

        Args:
            iamge_path (str): Path to the input image
        
        Returns:
            Image.Image: Loaded Pil image in RGB mode
        
        Raises:
            FileNotFoundError: If the image file does not exist
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        return image