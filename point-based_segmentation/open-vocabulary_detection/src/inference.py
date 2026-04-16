import torch
from torchvision.ops import nms

from src.utils.image_io import load_image, resize_image


def run_inference(
    model,
    processor,
    device,
    image_path,
    text_queries,
    score_threshold,
    nms_iou_threshold,
    max_size=1024,
):
    """
    Run open-vocabulary detection with OWLv2 / OWL-ViT.

    Steps:
    1. Load image
    2. Optionally resize image while preserving aspect ratio
    3. Prepare batched text labels
    4. Run model inference
    5. Post-process predictions
    6. Apply NMS
    7. Return a visualization-friendly result dictionary
    """
    print(f"Loading image: {image_path}")
    image = load_image(image_path)

    # Resize large images to improve speed and stability.
    image = resize_image(image, max_size=max_size)

    # OWL-family models expect batched text labels.
    # Example: [["pen", "laptop"]]
    text_labels = [text_queries]
    print(f"Text labels: {text_labels}")

    inputs = processor(
        text=text_labels,
        images=image,
        return_tensors="pt",
    )

    # Move tensors to the selected device.
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Running inference...")

    with torch.no_grad():
        outputs = model(**inputs)

    # target_sizes expects (height, width)
    target_sizes = torch.tensor([(image.height, image.width)], device=device)

    # For OWLv2, Hugging Face docs use:
    # processor.post_process_grounded_object_detection(
    #     outputs=outputs,
    #     target_sizes=target_sizes,
    #     threshold=...,
    #     text_labels=text_labels
    # )
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=score_threshold,
        text_labels=text_labels,
    )

    result = results[0]

    boxes = result["boxes"]
    scores = result["scores"]
    text_label_names = result["text_labels"]

    print(f"Raw detections: {len(boxes)}")

    # Apply NMS to reduce duplicate boxes.
    if len(boxes) > 0:
        keep = nms(boxes, scores, nms_iou_threshold)

        boxes = boxes[keep]
        scores = scores[keep]

        keep_list = keep.tolist() if hasattr(keep, "tolist") else list(keep)
        text_label_names = [text_label_names[i] for i in keep_list]

    filtered_result = {
        "boxes": boxes,
        "scores": scores,
        "text_labels": text_label_names,
    }

    print(f"Detections after NMS: {len(filtered_result['boxes'])}")

    return image, filtered_result