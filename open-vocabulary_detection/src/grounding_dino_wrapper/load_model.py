from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.utils.device import select_device

def load_model(model_id: str, device: str = "auto"):
    """
    Load MM Grounding DINO and processor

    Args:
        model_id (str): Hugging Face model ID
        device (str): "auto", "cuda", "mps", "cpu"
    
    Returns:
        tuple:
            model
            processor
            resolved_device (str)
    """
    device = select_device(device)

    print(f"Loading Model: {model_id}")
    print(f"Selected Device: {device}")

    # processor: image preprocessing, text tokenization, model input tensor generation
    processor = AutoProcessor.from_pretrained(model_id)

    # model: zero-shot / open-vocabulary detection
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    
    model.to(device)
    model.eval()

    return model, processor, device