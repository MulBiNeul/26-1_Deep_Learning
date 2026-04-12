import torch


def get_device(requested_device: str = "auto") -> str:
    """
    Select an available computation device.

    Args:
        requested_device (str): Preferred device ('auto', 'cuda', 'mps', 'cpu')

    Returns:
        str: Selected device name
    """

    requested_device = requested_device.lower()

    if requested_device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA requested but not available.")

    if requested_device == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("MPS requested but not available.")

    if requested_device == "cpu":
        return "cpu"

    if requested_device != "auto":
        raise ValueError(f"Unsupported device option: {requested_device}")

    # Auto selection: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"