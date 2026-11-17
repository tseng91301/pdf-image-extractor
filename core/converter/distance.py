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
        dx = max(x2_min - x1_max, x1_min - x2_max, 0.0)
        dy = max(y2_min - y1_max, y1_min - y2_max, 0.0)
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


def box_iou(box1: list, box2: list) -> float:
    """計算兩方框的交併比 (Intersection over Union)。"""
    b1 = normalize_box(box1)
    b2 = normalize_box(box2)

    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0.0 else 0.0


def horizontal_overlap_ratio(box1: list, box2: list) -> float:
    """
    計算 box1 與 box2 在水平 (X 軸) 上的交集寬度相對於較窄 box 的比例 (IoU-X-min)。
    若值 >= 0.25，表示在同一欄或大幅重疊；若為 0.0，表示在水平方向完全錯開（跨欄）。
    """
    b1 = normalize_box(box1)
    b2 = normalize_box(box2)

    x1_min, _, x1_max, _ = b1
    x2_min, _, x2_max, _ = b2

    overlap_x = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    w1 = max(1e-6, x1_max - x1_min)
    w2 = max(1e-6, x2_max - x2_min)
    return overlap_x / min(w1, w2)


def column_aware_distance(img_box: list, text_box: list, col_penalty: float = 3.0) -> float:
    """
    計算帶有欄位感知 (Column-aware) 的距離度量。
    同欄的段落 (水平重疊率 >= 0.25) 計算實際歐氏距離。
    跨欄或無水平重疊的段落 (水平錯開，dx > 0) 施加 col_penalty (預設 3.0 倍懲罰)，
    避免在多欄式 (Multi-column) 排版中誤抓相鄰欄位的無關文字。
    """
    b1 = normalize_box(img_box)
    b2 = normalize_box(text_box)

    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2

    dx = max(x2_min - x1_max, x1_min - x2_max, 0.0)
    dy = max(y2_min - y1_max, y1_min - y2_max, 0.0)

    overlap = horizontal_overlap_ratio(img_box, text_box)
    if overlap >= 0.25:
        # 同欄位或高重疊：正常計算距離
        return math.hypot(dx, dy)
    else:
        # 跨欄或無水平重疊：對水平距離施加顯著懲罰
        return math.hypot(dx * col_penalty, dy)
