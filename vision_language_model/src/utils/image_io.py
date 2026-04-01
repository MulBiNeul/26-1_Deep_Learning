from pathlib import Path
from PIL import Image

class ImageLoader:
    """ Load images for vision-language inference """

    @staticmethod
    def load_image(image_path: str, max_size: int = 512) -> Image.Image:
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
        width, height = image.size
        long_side = max(width, height)

        if long_side > max_size:
            scale = max_size / long_side
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image