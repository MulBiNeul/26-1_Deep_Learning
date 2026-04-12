def validate_points_and_labels(points, labels):
    """
    Validate point coordinates and corresponding labels.

    Args:
        points (list): List of point coordinates in [x, y] format
        labels (list): List of labels for each point (1 for foreground, 0 for background)

    Raises:
        ValueError: If points or labels are invalid
    """

    if not isinstance(points, list) or len(points) == 0:
        raise ValueError("points must be a non-empty list.")

    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError("labels must be a non-empty list.")

    if len(points) != len(labels):
        raise ValueError("The number of points and labels must match.")

    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("Each point must be in [x, y] format.")
        if not all(isinstance(value, (int, float)) for value in point):
            raise ValueError("Point coordinates must be numeric values.")

    for label in labels:
        if label not in [0, 1]:
            raise ValueError("Each label must be 0 (background) or 1 (foreground).")


def scale_points(points, scale):
    """
    Scale point coordinates according to the image resize ratio.

    Args:
        points (list): List of point coordinates in [x, y] format
        scale (float): Resize scale factor

    Returns:
        list: Scaled point coordinates
    """

    if scale == 1.0:
        return points

    scaled_points = []
    for x, y in points:
        scaled_x = int(round(x * scale))
        scaled_y = int(round(y * scale))
        scaled_points.append([scaled_x, scaled_y])

    return scaled_points


def build_sam_inputs(points, labels):
    """
    Convert points and labels into the format required by the SAM processor.

    SAM processor format:
        input_points: [[[x1, y1], [x2, y2], ...]]
        input_labels: [[1, 0, ...]]

    Args:
        points (list): List of point coordinates
        labels (list): List of point labels

    Returns:
        tuple: Formatted input_points and input_labels
    """

    validate_points_and_labels(points, labels)

    input_points = [points]
    input_labels = [labels]

    return input_points, input_labels