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
    print(f"Loading image: {image_path}")

    # RGB unification
    image = Image.open(image_path).convert("RGB")

    # Convert text prompt to input prompt for model
    text_prompt = build_text_prompt(text_queries)
    print(f"text prompt: {text_prompt}")

    # Convert text and image to model input tensor using processor
    input = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt"
    )

    # Move all generated tensors to the selected device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Running inference...")

    # Gradient is not needed during inference
    with torch.no_grad():
        outputs = model(**inputs)

    # model output box is generally regularization coordinate,
    # target size is needed for reverting real image size
    # PIL image.size: (width, height) -> target_sizes: (height, width)
    target_sizes = torch.tensor([image.size[::-1]], device=device)

    # post-processing:
    # converting regularization box to real pixel coordinate
    # applying threshold
    # mapping label
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )

    result = results[0]

    print(f"detection: {len(result['boxes'])}")

    return image, result