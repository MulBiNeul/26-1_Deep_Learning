import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config
from src.utils.image_io import save_image
from src.utils.visualization import draw_boxes

from src.predictor import get_text_queries
from src.inference import run_inference
from src.owl_wrapper.load_model import load_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Open-vocabulary detection with OWL-ViT / OWLv2"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help='Comma-separated text queries, e.g. "pen, laptop"',
    )
    return parser.parse_args()


def main():
    # 1. Load config
    args = parse_args()
    cfg = load_config(args.config)

    # 2. Get text queries
    text_queries = get_text_queries(args.text)

    # 3. Load model
    model, processor, device = load_model(
        model_name=cfg["model"]["model_name"],
        model_id=cfg["model"]["model_id"],
        device=cfg["model"]["device"],
    )

    # 4. Run inference
    image, result = run_inference(
        model=model,
        processor=processor,
        device=device,
        image_path=cfg["input"]["image_path"],
        text_queries=text_queries,
        score_threshold=cfg["thresholds"]["score_threshold"],
        nms_iou_threshold=cfg["thresholds"]["nms_iou_threshold"],
        max_size=cfg["preprocess"]["max_size"],
    )

    # 5. Draw boxes
    vis_image = draw_boxes(image, result)

    # 6. Save output
    save_image(
        image=vis_image,
        save_dir=cfg["output"]["save_dir"],
        filename=cfg["output"]["save_image_name"],
    )


if __name__ == "__main__":
    main()