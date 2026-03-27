import argparse
from pathlib import Path
import yaml

from src.inference import run_inference


def load_config(config_path: str) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Point-based segmentation with SAM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML 설정 파일 경로"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_inference(config)


if __name__ == "__main__":
    main()