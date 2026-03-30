from pathlib import Path
from transformers import SamModel, SamProcessor


MODEL_NAME = "facebook/sam-vit-base"
SAVE_DIR = Path("checkpoints") / "sam-vit-base"


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Hugging Face에서 모델 다운로드: {MODEL_NAME}")
    processor = SamProcessor.from_pretrained(MODEL_NAME)
    model = SamModel.from_pretrained(MODEL_NAME)

    print(f"[2/3] 로컬에 저장: {SAVE_DIR}")
    processor.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)

    print("[3/3] 완료")
    print(f"저장 위치: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()