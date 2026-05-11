"""
Удаление существующих объектов с кадра по bbox (маска + cv2.inpaint)
для cut-paste аугментации с согласованной YOLO-разметкой (только новые объекты).
"""
import cv2
import numpy as np


def remove_labeled_instances_bgr(
    image_bgr,
    objects_list,
    dilate=2,
    inpaint_radius=3,
    max_mask_fraction=0.85,
):
    """
    objects_list: элементы вида (cls, x1, y1, x2, y2) в пикселях, как в analyze_dataset.
    Возвращает BGR uint8; при пустом списке или огромной маске — копию исходника.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    out = image_bgr.copy()
    if not objects_list:
        return out

    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for item in objects_list:
        if item is None:
            continue
        if len(item) >= 5:
            _cls, x1, y1, x2, y2 = item[0], int(item[1]), int(item[2]), int(item[3]), int(item[4])
        elif len(item) >= 4:
            x1, y1, x2, y2 = int(item[0]), int(item[1]), int(item[2]), int(item[3])
        else:
            continue
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 255

    if dilate > 0 and np.any(mask):
        k = max(1, int(dilate)) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)

    frac = float(np.count_nonzero(mask)) / float(h * w + 1e-9)
    if frac <= 1e-6:
        return out
    if frac > max_mask_fraction:
        return out

    inpainted = cv2.inpaint(out, mask, int(max(1, inpaint_radius)), cv2.INPAINT_TELEA)
    return inpainted
