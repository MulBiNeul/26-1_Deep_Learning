import torch

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
    if device != "auto":
        return device

    cuda_built = torch.version.cuda is not None
    cuda_available = torch.cuda.is_available()

    if cuda_built and cuda_available:
        try:
            _ = torch.tensor([0.0]).to("cuda")
            return "cuda"
        except Exception:
            pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"