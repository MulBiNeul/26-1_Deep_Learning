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
    # Specific Case: User selected
    if device != "auto":
        return device
    
    # Following priority (CUDA > MPS > CPU)
    if torch.cuda.is_available:
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"