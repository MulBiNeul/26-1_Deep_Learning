import torch
from PIL import Image

def build_text_prompt(text_queries: list[str]) -> str:
    """
    Converts the list of queries entered by the user into a text prompt format suitable for inserting into Grounding DINO
    
    Example:
        ["pen", "laptop"] -> "pen. laptop."

    Args:
        text_queries (list[str]): the list of queries which user wants to detect
    
    Returns:
        str: Prompt input for Grounding DINO
    """
    # Remove blank + Remove vaccant str
    cleaned_queries = [q.strip() for q in text_queries if q.strip()]

    # Make exception if there isn't any query
    if not cleaned_queries:
        raise ValueError("Text query is empty. Please provice at least one query.")
    
    # Sentence input is general to Grounding DINO.
    # Each query is connected by dot (.)
    return ". ".join(cleaned_queries) + "."

def run_inference(
        model,
        processor,
        device: str,
        image_path: str,
        text_queries: list[str],
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
):
    """
    Execute the MM Grounding DINO inference
    
    Args:
        model:
            Hugging Face zero-shot object detection model
        processor:
            Hugging FAce processor
        device (str):
            "cuda", "mps", "cpu"
        image_path (str):
            path of image input
        text_queries (list[str]):
            example: ["pen", "laptop"]
        box_threshold (float):
            box confidence threshold
        text_threshold (float):
            text-matching threshold
    
    Returns:
        tuple:
            image (PIL.Image.Image): raw image
            result (dict): detection result of first image
    """
    print(f"[INFO] loading image: {image_path}")

    # RGB unification
    image = Image.open(image_path).convert("RGB")