import torch


def get_device(requested_device: str = "auto") -> str:
    requested_device = requested_device.lower()

    if requested_device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA를 요청했지만 현재 사용할 수 없습니다.")

    if requested_device == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("MPS를 요청했지만 현재 사용할 수 없습니다.")

    if requested_device == "cpu":
        return "cpu"

    if requested_device != "auto":
        raise ValueError(f"지원하지 않는 device 옵션입니다: {requested_device}")

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"