from pathlib import Path

from src.sam_wrapper.load_model import load_sam_model
from src.utils.image_io import load_image, resize_image, ensure_dir
from src.utils.points import scale_points
from src.predictor import predict_mask
from src.utils.visualization import (
    save_mask,
    save_prompt_image,
    save_overlay,
    save_three_panel_figure,
)


def run_inference(config: dict):
    """
    Run point-based segmentation using SAM with a given configuration.

    Pipeline:
        1. Load model
        2. Load image
        3. (Optional) Resize image and scale points
        4. Run SAM inference
        5. Save results (mask, prompt, overlay, panel)

    Args:
        config (dict): Configuration dictionary loaded from YAML file
    """

    model_dir = config["model"]["local_dir"]
    requested_device = config["runtime"]["device"]

    image_path = config["input"]["image_path"]
    output_dir = config["output"]["dir"]

    points = config["prompt"]["points"]
    labels = config["prompt"]["labels"]

    resize_cfg = config["input"].get("resize", {})
    resize_enabled = resize_cfg.get("enabled", False)
    resize_size = resize_cfg.get("size", 1024)

    save_mask_flag = config["output"].get("save_mask", True)
    save_overlay_flag = config["output"].get("save_overlay", True)
    save_panel_flag = config["output"].get("save_panel", True)

    ensure_dir(output_dir)

    print("[1/5] Loading SAM model...")
    processor, model, device = load_sam_model(model_dir, requested_device)

    print(f"[2/5] Loading image... ({image_path})")
    image = load_image(image_path)

    if resize_enabled:
        print(f"[2-1/5] Applying resize (long side -> {resize_size})")
        image, scale = resize_image(image, resize_size)
        points = scale_points(points, scale)
        print(f"Resize scale: {scale:.4f}")
        print(f"Scaled points: {points}")

    print(f"[3/5] Running inference... (device={device})")
    result = predict_mask(
        image=image,
        processor=processor,
        model=model,
        device=device,
        points=points,
        labels=labels,
    )

    image_stem = Path(image_path).stem
    mask_path = Path(output_dir) / f"{image_stem}_mask.png"
    prompt_path = Path(output_dir) / f"{image_stem}_prompt.png"
    overlay_path = Path(output_dir) / f"{image_stem}_overlay.png"
    panel_path = Path(output_dir) / f"{image_stem}_panel.png"

    print("[4/5] Saving results...")
    if save_mask_flag:
        save_mask(result["mask"], str(mask_path))

    save_prompt_image(image, points, labels, str(prompt_path))

    if save_overlay_flag:
        save_overlay(image, result["mask"], str(overlay_path))

    if save_panel_flag:
        save_three_panel_figure(
            original_image=image,
            prompt_image_path=str(prompt_path),
            overlay_image_path=str(overlay_path),
            save_path=str(panel_path),
        )

    print("[5/5] Done")
    print(f"Best score: {result['score']:.4f}")
    print(f"Prompt saved: {prompt_path}")

    if save_mask_flag:
        print(f"Mask saved: {mask_path}")
    if save_overlay_flag:
        print(f"Overlay saved: {overlay_path}")
    if save_panel_flag:
        print(f"Panel saved: {panel_path}")