import torch


def select_device(device: str = "auto") -> str:
    """
    Select the appropriate device.

    Priority:
    1. CUDA (Windows / Linux + NVIDIA GPU)
    2. MPS (macOS Apple Silicon)
    3. CPU

    Args:
        device (str): "auto", "cuda", "mps", "cpu"

    Returns:
        str: Final device string
    """

    # User-specified device
    if device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available. Falling back to CPU.")
            return "cpu"

        if device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                print("[WARNING] MPS requested but not available. Falling back to CPU.")
                return "cpu"

        print(f"[INFO] Using user-specified device: {device}")
        return device

    # Auto selection
    cuda_built = torch.version.cuda is not None
    cuda_available = torch.cuda.is_available()

    if cuda_built and cuda_available:
        try:
            _ = torch.tensor([0.0]).to("cuda")
            print("[INFO] Using CUDA device")
            return "cuda"
        except Exception:
            pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] Using MPS device")
        return "mps"

    print("[INFO] Using CPU")
    return "cpu"