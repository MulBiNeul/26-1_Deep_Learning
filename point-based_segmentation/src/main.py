import argparse
import yaml
from pathlib import Path

from src.interactive import run_interactive
from src.inference import run_inference


def load_config(config_path: str):
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, default="interactive",
                        choices=["interactive", "inference"])

    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "interactive":
        run_interactive(config)
    else:
        run_inference(config)


if __name__ == "__main__":
    main()

# python -m src.main --mode interactive
# python -m src.main --mode inference