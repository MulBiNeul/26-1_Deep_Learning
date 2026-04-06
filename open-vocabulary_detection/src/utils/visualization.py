from PIL import Image, ImageDraw

def draw_boxes(image, result):
    """
    Draw detection results on the raw image

    Args:
        image:
            PIL image object
        
        result (dict):
            detection result including following keys
            - "boxes"
            - "scores"
            - "labels"

    Returns:
        image:
            PIL.image with bounding box and label
    """
    draw = ImageDraw.Draw(image)

    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    text_labels = result.get("text_labels", [])

    # Traversal each detected object
    for box, score, label in zip(boxes, scores, text_labels):
        # tensor -> python value
        x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
        score_value = float(score.item()) if hasattr(score, "item") else float(score)
        
        # draw text
        # label could be criteria of Grounding DINO post-process result
        label_text = str(label)

        # draw box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=3,
        )

        # display the text on upper left of the box
        draw.text(
            (x1, max(0, y1- 12)),
            f"{label_text}: {score_value:.2f}",
            fill="red",
        )

    return image