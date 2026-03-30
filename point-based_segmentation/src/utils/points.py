def validate_points_and_labels(points, labels):
    if not isinstance(points, list) or len(points) == 0:
        raise ValueError("points는 비어 있지 않은 리스트여야 합니다.")

    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError("labels는 비어 있지 않은 리스트여야 합니다.")

    if len(points) != len(labels):
        raise ValueError("points 개수와 labels 개수가 일치해야 합니다.")

    for p in points:
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError("각 point는 [x, y] 형식이어야 합니다.")
        if not all(isinstance(v, (int, float)) for v in p):
            raise ValueError("point 좌표는 숫자여야 합니다.")

    for lb in labels:
        if lb not in [0, 1]:
            raise ValueError("label은 0(background) 또는 1(foreground)이어야 합니다.")


def scale_points(points, scale):
    """
    resize 비율에 맞게 point 좌표를 함께 변환
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
    validate_points_and_labels(points, labels)
    """
    SAM processor format:
    input_points: [[[x1, y1], [x2, y2], ...]]
    input_labels: [[1, 0, ...]]
    """
    input_points = [points]
    input_labels = [labels]

    return input_points, input_labels