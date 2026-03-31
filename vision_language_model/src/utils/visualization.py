import cv2
import numpy as np
import textwrap

class Visualizer:
    """ Enhanced visualization for VLM results with text wrapping """

    @staticmethod
    def wrap_text(text: str, max_chars: int = 60) -> list[str]:
        """
        Wrap text into multiple lines

        Args:
            text (str): Input text
            max_chars (int): Maximum characters per line
        
        Returns:
            list[str]: Wrapped text lines
        """
        return textwrap.wrap(text, width=max_chars)
    
    @staticmethod
    def show_image_with_text(image, text: str) -> None:
        """
        Display image with multi-line text overlay

        Args:
            image: PIL image
            text (str): Text to display
        """
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize if the image is too large for display
        max_width = 800
        if img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        lines = Visualizer.wrap_text(text)
        line_height = 30
        padding = 20
        box_height = line_height * len(lines) + padding