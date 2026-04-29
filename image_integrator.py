import cv2
import numpy as np
import random
import math


def adjust_brightness_contrast(obj_bgr, target_mean, target_std):
    obj_mean, obj_std = cv2.meanStdDev(obj_bgr)
    obj_mean = np.array(obj_mean).flatten()
    obj_std = np.array(obj_std).flatten()
    scale = np.divide(target_std, obj_std, out=np.ones_like(target_std), where=obj_std != 0)
    adjusted = (obj_bgr.astype(np.float32) - obj_mean) * scale + target_mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def create_feather_mask(h, w, border_ratio=0.2):
    y, x = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    max_dist = max(cy, cx) * (1.0 - border_ratio)
    dist_to_edge = np.minimum(np.minimum(y, h - 1 - y), np.minimum(x, w - 1 - x)).astype(np.float32)
    mask = np.clip(dist_to_edge / (max_dist + 1e-6), 0.0, 1.0)
    return mask


def insert_object(
    background,
    object_img,
    mask=None,
    position=None,
    scale_range=(0.1, 0.3),      # доля от меньшей стороны фона
    angle_range=(-30, 30),
    color_adapt=True,
    blend_strength=0.5
):
    bg_h, bg_w = background.shape[:2]
    obj_h, obj_w = object_img.shape[:2]

    # Определяем масштаб относительно фона
    min_side = min(bg_h, bg_w)
    scale = random.uniform(*scale_range) * min_side / max(obj_h, obj_w)

    angle = random.uniform(*angle_range)
    if position is None:
        cx = random.randint(0, max(1, bg_w - 1))
        cy = random.randint(0, max(1, bg_h - 1))
    else:
        cx, cy = position

    # Адаптация цвета ко всему фону (быстрее и стабильнее)
    if color_adapt:
        bg_mean, bg_std = cv2.meanStdDev(background)
        bg_mean = np.array(bg_mean).flatten()
        bg_std = np.array(bg_std).flatten()
        object_img = adjust_brightness_contrast(object_img, bg_mean, bg_std)

    # Аффинное преобразование
    M = cv2.getRotationMatrix2D((obj_w / 2, obj_h / 2), angle, scale)
    M[0, 2] += cx - obj_w / 2 * scale
    M[1, 2] += cy - obj_h / 2 * scale

    warped_obj = cv2.warpAffine(object_img, M, (bg_w, bg_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

    if mask is None:
        mask = create_feather_mask(obj_h, obj_w)
    warped_mask = cv2.warpAffine(mask, M, (bg_w, bg_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)

    # Регулируем силу смешивания
    if blend_strength >= 1.0:
        effective_mask = warped_mask
    else:
        effective_mask = warped_mask * blend_strength + (1.0 - blend_strength) * (warped_mask > 0.1).astype(np.float32)

    effective_mask_3ch = np.dstack([effective_mask] * 3)
    result = (warped_obj * effective_mask_3ch + background * (1 - effective_mask_3ch)).astype(np.uint8)

    # Bounding box по маске (уверенный порог 0.2)
    ys, xs = np.where(warped_mask > 0.2)
    if len(xs) == 0:
        # fallback
        x_center, y_center = cx / bg_w, cy / bg_h
        bw, bh = 0.05, 0.05
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_center = ((x_min + x_max) / 2) / bg_w
        y_center = ((y_min + y_max) / 2) / bg_h
        bw = (x_max - x_min) / bg_w
        bh = (y_max - y_min) / bg_h

    return result, (x_center, y_center, bw, bh)


def match_object_to_background(obj_img, background):
    """Адаптирует яркость/контраст объекта под всё фоновое изображение."""
    bg_mean, bg_std = cv2.meanStdDev(background)
    target_mean = np.array(bg_mean).flatten()
    target_std = np.array(bg_std).flatten()
    return adjust_brightness_contrast(obj_img, target_mean, target_std)