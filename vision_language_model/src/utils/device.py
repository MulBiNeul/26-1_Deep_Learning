import torch

class DeviceManager:
    """ Manage device selectionfor model inference"""

    @staticmethod
    def get_device(device_option: str) -> str:
        """
        Resolve the runtime device based on the user option.

        Args:
            device_option (str): Device option from config.
                                 Supported values: auto, cpu, cuda, mps

        Returns:
            str: Resolved device string

        Raisees:
            ValueError: If the requested device option is invalid or unavailable                   
        """
        device_option = device_option.lower() # Normalize input for case-insensitive comparison

        if device_option == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        
        if device_option == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise ValueError("CUDA is not available on this system.")
        if device_option == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            raise ValueError("MPS is not available on this system.")
        if device_option == "cpu":
            return "cpu"

        raise ValueError(f"Invalid device option: {device_option}")