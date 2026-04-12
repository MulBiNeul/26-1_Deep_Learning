from pathlib import Path
from transformers import SamModel, SamProcessor


MODEL_NAME = "facebook/sam-vit-base"
SAVE_DIR = Path("checkpoints") / "sam-vit-base"


def main():
    """
    Download SAM model and processor from Hugging Face
    and save them locally for offline use.

    The saved model can be reused without requiring
    an internet connection or Hugging Face authentication.
    """

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Downloading model from Hugging Face: {MODEL_NAME}")
    processor = SamProcessor.from_pretrained(MODEL_NAME)
    model = SamModel.from_pretrained(MODEL_NAME)

    print(f"[2/3] Saving locally: {SAVE_DIR}")
    processor.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    print("[3/3] Done")
    print(f"Saved at: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()