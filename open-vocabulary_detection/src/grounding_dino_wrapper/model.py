import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def select_device(device: str = "auto") -> str:
    """
    Select the appropriate device

    Priority:
    1. CUDA (Windows / Linux + NVIDA GPU)
    2. MPS (MacOS Apple Silicon)
    3. CPU

    Args:
        device (str): "auto", "cuda", "mps", "cpu"
    
    Returns:
        str: Final device string to be used
    """
    # Specific Case: User selected
    if device != "auto":
        return device
    
    # Following priority (CUDA > MPS > CPU)
    if torch.cuda.is_available:
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GroudingDINOModel:
    
    """
    Wrapper Class:
        Managing and loading MM Grounding DINO model, processor
    """

    def __init__(self, model_id: str, device: str = "auto"):

        """
        Args:
            model_id (str): Hugging Face model ID
                ex) "openmmlab-community/mm_grounding_dino_large_all"

            device (str): "auto", "cuda", "mps", "cpu"
        """

        self.model_id = model_id
        self.device = select_device(device)

        print(f"model_id: {self.model_id}")
        print(f"selected device: {self.device}")

        # AutoProcessor:
        # - Image Preprocessing
        # - Text Tokenization
        # - Model Input Tensor Generation
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # AutoModelForZeroShotObjectDetection:
        # - Load open-vocabulary / zero-shot object detection model
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        )

        # Move model to selected device
        self.model.to(self.device)

        # Inference Mode
        self.model.eval()

        def get_model(self):
            """
            Return the loaded model
            """
            return self.model
        
        def get_processor(self):
            """
            Return the loaded processor
            """
            return self.processor
        
        def get_device(self):
            """
            Return the selected device
            """
            return self.device