import math

def normalize_box(box):
    return [float(v) for v in box]

def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def box_distance(box1: list, box2: list, type="avg"):
    box1 = normalize_box(box1)
    box2 = normalize_box(box2)

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    if type == "avg":
        c1x, c1y = box_center(box1)
        c2x, c2y = box_center(box2)
        return math.hypot(c1x - c2x, c1y - c2y)

    elif type == "min":
        dx = max(x2_min - x1_max, x1_min - x2_max, 0)
        dy = max(y2_min - y1_max, y1_min - y2_max, 0)
        return math.hypot(dx, dy)

    elif type == "max":
        corners1 = [
            (x1_min, y1_min),
            (x1_min, y1_max),
            (x1_max, y1_min),
            (x1_max, y1_max),
        ]
        corners2 = [
            (x2_min, y2_min),
            (x2_min, y2_max),
            (x2_max, y2_min),
            (x2_max, y2_max),
        ]
        return max(
            math.hypot(x1 - x2, y1 - y2)
            for (x1, y1) in corners1
            for (x2, y2) in corners2
        )

    else:
        raise ValueError(f"Unknown distance type: {type}")
