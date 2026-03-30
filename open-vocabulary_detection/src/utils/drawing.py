from PIL import ImageDraw

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

    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]

    # Traversal each detected object
    for box, score, label in zip(boxes, scores, labels):
        # tensor -> python value
        x1, y1, x2, y2 = [round(v, 2) for v in box.tolist()]
        score_value = float(score.item())

        # draw box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=3,
        )

        # draw text
        # label could be criteria of Grounding DINO post-process result
        text = f"{label}: {score_value:.2f}"

        # display the text on upper left of the box
        draw.txt((x1, y1), text, fill="red")

    return image