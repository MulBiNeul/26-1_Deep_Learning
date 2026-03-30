from src.utils.device import select_device
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)


def load_model(model_name: str, model_id: str, device: str = "auto"):
    """
    Load an OWL-family model and processor.

    Supported model_name values:
        - "owlvit"
        - "owlv2"
    """
    resolved_device = select_device(device)

    print(f"Loading Model: {model_id}")
    print(f"Selected Device: {resolved_device}")

    if model_name.lower() == "owlvit":
        processor = OwlViTProcessor.from_pretrained(model_id)
        model = OwlViTForObjectDetection.from_pretrained(model_id)
    elif model_name.lower() == "owlv2":
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)
    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Use 'owlvit' or 'owlv2'."
        )

    model.to(resolved_device)
    model.eval()

    return model, processor, resolved_device