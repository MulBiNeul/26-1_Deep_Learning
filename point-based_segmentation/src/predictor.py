import torch
import numpy as np

from src.utils.points import build_sam_inputs


def predict_mask(image, processor, model, device, points, labels):
    input_points, input_labels = build_sam_inputs(points, labels)

    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    )

    original_sizes = inputs["original_sizes"]
    reshaped_input_sizes = inputs["reshaped_input_sizes"]

    converted_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.dtype == torch.float64:
                v = v.float()
            converted_inputs[k] = v.to(device)
        else:
            converted_inputs[k] = v

    inputs = converted_inputs

    with torch.no_grad():
        outputs = model(
            **inputs,
            multimask_output=False
        )

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=original_sizes,
        reshaped_input_sizes=reshaped_input_sizes
    )

    print("post_process_masks len:", len(masks))
    print("post_process_masks[0].shape:", masks[0].shape)

    mask_tensor = masks[0]

    # 모든 singleton 차원 제거
    mask_tensor = mask_tensor.squeeze()

    print("squeezed mask shape:", mask_tensor.shape)

    if mask_tensor.ndim != 2:
        raise ValueError(f"예상하지 못한 mask shape: {mask_tensor.shape}")

    score_tensor = outputs.iou_scores.squeeze()

    mask = mask_tensor.numpy()
    mask = (mask > 0).astype(np.uint8)

    score = float(score_tensor.item())

    return {
        "mask": mask,
        "score": score
    }