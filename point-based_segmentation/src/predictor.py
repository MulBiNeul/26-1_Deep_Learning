import torch
import numpy as np

from src.utils.points import build_sam_inputs


def predict_mask(image, processor, model, device, points, labels):
    """
    Perform point-based segmentation using SAM.

    Pipeline:
        1. Convert points to SAM input format
        2. Preprocess inputs with processor
        3. Move inputs to device (with dtype handling)
        4. Run model inference
        5. Post-process masks to original image size
        6. Select best mask and convert to binary

    Args:
        image (PIL.Image): Input image
        processor: SAM processor
        model: SAM model
        device (str): Target device (cpu, cuda, mps)
        points (list): List of point coordinates
        labels (list): List of labels (1=foreground, 0=background)

    Returns:
        dict: {
            "mask": np.ndarray (H, W),
            "score": float
        }
    """

    # Step 1: Build SAM input format
    input_points, input_labels = build_sam_inputs(points, labels)

    # Step 2: Preprocess inputs
    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    )

    original_sizes = inputs["original_sizes"]
    reshaped_input_sizes = inputs["reshaped_input_sizes"]

    # Step 3: Move to device with dtype fix (MPS compatibility)
    converted_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.dtype == torch.float64:
                value = value.float()  # Convert to float32
            converted_inputs[key] = value.to(device)
        else:
            converted_inputs[key] = value

    inputs = converted_inputs

    # Step 4: Run inference
    with torch.no_grad():
        outputs = model(
            **inputs,
            multimask_output=False
        )

    # Step 5: Post-process masks to original size
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=original_sizes,
        reshaped_input_sizes=reshaped_input_sizes
    )

    # Step 6: Extract and clean mask
    mask_tensor = masks[0].squeeze()

    if mask_tensor.ndim != 2:
        raise ValueError(f"Unexpected mask shape: {mask_tensor.shape}")

    # Convert to binary mask
    mask = mask_tensor.numpy()
    mask = (mask > 0).astype(np.uint8)

    # Extract score
    score = float(outputs.iou_scores.squeeze().item())

    return {
        "mask": mask,
        "score": score
    }