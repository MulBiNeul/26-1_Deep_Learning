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
    def close() -> None:
        """ Close the visualization window """
        cv2.destroyAllWindows()