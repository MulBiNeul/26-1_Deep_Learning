import torch
from torchvision.ops import nms

from src.utils.image_io import load_image


def run_inference(
    model,
    processor,
    device,
    image_path,
    text_queries,
    score_threshold,
    nms_iou_threshold,
):
    """
    Run open-vocabulary detection with OWLv2 / OWL-ViT.

    Steps:
    1. Load image
    2. Prepare text labels
    3. Run model inference
    4. Post-process predictions
    5. Apply NMS
    6. Return a visualization-friendly result dictionary
    """
    print(f"Loading image: {image_path}")
    image = load_image(image_path)

    # OWLv2 expects batched text labels.
    # Example: [["pen", "laptop"]]
    text_labels = [text_queries]
    print(f"Text labels: {text_labels}")

    inputs = processor(
        text=text_labels,
        images=image,
        return_tensors="pt",
    )

    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Running inference...")

    with torch.no_grad():
        outputs = model(**inputs)

    # target_sizes expects (height, width)
    target_sizes = torch.tensor([(image.height, image.width)], device=device)

    # OWLv2 official docs use post_process_grounded_object_detection
    # and pass text_labels to recover string labels.
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

    # Apply NMS to reduce duplicate boxes
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