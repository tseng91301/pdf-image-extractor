def map_bbox(
    x1, y1, x2, y2,
    W_old, H_old,
    W_new, H_new
):
    sx = W_new / W_old
    sy = H_new / H_old

    return (
        x1 * sx,
        y1 * sy,
        x2 * sx,
        y2 * sy
    )