import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

class QwenModelLoader:
    """ Load the Qwen vision-language model and processor """

    def __init__(self, model_name: str, device: str):
        """
        Initialize the model loader
        
        Args:
            model_name (str): Hugging Face model name
            device (str): Runtime device string
        """
        self.model_name = model_name
        self.device = device

    def load(self):
        """
        Load the model and processor

        Returns:
            tuple: (model, processor)
        """
        processor = AutoProcessor.from_pretrained(self.model_name)

        if self.device in ["cuda", "mps"]:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=None,
            )
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=None,
            )

        model = model.to(self.device)
        model.eval()

        return model, processor