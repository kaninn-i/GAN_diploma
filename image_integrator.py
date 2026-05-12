"""
Интеграция объектов в фоновые изображения.

Основной метод — Пуассоново смешивание (cv2.seamlessClone / MIXED_CLONE),
которое убирает «вырезанный» вид объекта.
При ошибках или граничных случаях — фолбэк на простое копирование с маской.
"""

import cv2
import numpy as np
import random


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def adjust_brightness_contrast(obj_bgr, target_mean, target_std):
    """Адаптирует яркость/контраст объекта под статистику фона."""
    obj_mean, obj_std = cv2.meanStdDev(obj_bgr)
    obj_mean = np.array(obj_mean).flatten()
    obj_std  = np.array(obj_std).flatten()
    scale = np.divide(target_std, obj_std,
                      out=np.ones_like(target_std), where=obj_std != 0)
    adjusted = (obj_bgr.astype(np.float32) - obj_mean) * scale + target_mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def match_object_to_background(obj_img, background):
    """Адаптирует яркость/контраст объекта под всё фоновое изображение."""
    bg_mean, bg_std = cv2.meanStdDev(background)
    target_mean = np.array(bg_mean).flatten()
    target_std  = np.array(bg_std).flatten()
    return adjust_brightness_contrast(obj_img, target_mean, target_std)


def _ellipse_mask(h: int, w: int, shrink: float = 0.06) -> np.ndarray:
    """
    Белая эллиптическая маска (uint8) для seamlessClone.
    shrink — отступ от края (доля от размера), нужен для Пуассона.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    ax = max(1, int(cx * (1.0 - shrink)))
    ay = max(1, int(cy * (1.0 - shrink)))
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask


def _rotate_scale(img: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Вращение + масштабирование (preserve canvas size)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция вставки
# ─────────────────────────────────────────────────────────────────────────────

def insert_object(
    background: np.ndarray,
    object_img: np.ndarray,
    mask: np.ndarray | None = None,
    position: tuple | None = None,
    scale_range: tuple = (0.10, 0.30),   # доля от меньшей стороны фона
    angle_range: tuple = (-20, 20),
    color_adapt: bool = True,
    blend_strength: float = 0.7,         # 1.0 = полный Пуассон, <1 — ослабление
) -> tuple[np.ndarray, tuple]:
    """
    Вставляет object_img в background с помощью Пуассонова смешивания
    (cv2.seamlessClone MIXED_CLONE).

    Возвращает (result_bgr, (x_center, y_center, bw, bh)) в нормированных координатах.
    При неудаче — фолбэк на прямую вставку с маской.
    """
    bg_h, bg_w = background.shape[:2]
    obj_h, obj_w = object_img.shape[:2]

    # ── выбор масштаба и угла ──────────────────────────────────────────────
    min_side = min(bg_h, bg_w)
    raw_scale = random.uniform(*scale_range) * min_side / max(obj_h, obj_w)
    angle = random.uniform(*angle_range)

    # ── целевой размер объекта ─────────────────────────────────────────────
    new_w = max(8, int(obj_w * raw_scale))
    new_h = max(8, int(obj_h * raw_scale))

    # ── цветовая адаптация ─────────────────────────────────────────────────
    if color_adapt:
        object_img = match_object_to_background(object_img, background)

    # ── ресайз + вращение ──────────────────────────────────────────────────
    resized = cv2.resize(object_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if angle != 0:
        resized = _rotate_scale(resized, angle, 1.0)

    # ── центр вставки ──────────────────────────────────────────────────────
    half_w, half_h = new_w // 2, new_h // 2
    if position is None:
        # ограничиваем, чтобы seamlessClone не вылез за границы
        margin_x = half_w + 2
        margin_y = half_h + 2
        cx = random.randint(margin_x, max(margin_x, bg_w - margin_x))
        cy = random.randint(margin_y, max(margin_y, bg_h - margin_y))
    else:
        cx = int(np.clip(position[0], half_w + 2, bg_w - half_w - 2))
        cy = int(np.clip(position[1], half_h + 2, bg_h - half_h - 2))

    # ── маска для seamlessClone ─────────────────────────────────────────────
    if mask is None:
        obj_mask = _ellipse_mask(new_h, new_w, shrink=0.08)
    else:
        obj_mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST)
        obj_mask = np.clip(obj_mask, 0, 255).astype(np.uint8)

    # ── Пуассоново смешивание ──────────────────────────────────────────────
    result = _poisson_clone(resized, background, obj_mask, cx, cy, blend_strength)

    # ── bounding box ──────────────────────────────────────────────────────
    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(bg_w, cx + half_w)
    y2 = min(bg_h, cy + half_h)

    x_center = ((x1 + x2) / 2) / bg_w
    y_center = ((y1 + y2) / 2) / bg_h
    bw = (x2 - x1) / bg_w
    bh = (y2 - y1) / bg_h

    return result, (x_center, y_center, bw, bh)


# ─────────────────────────────────────────────────────────────────────────────
# Пуассонов клон с фолбэком
# ─────────────────────────────────────────────────────────────────────────────

def _poisson_clone(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
    blend_strength: float,
) -> np.ndarray:
    """
    Выполняет cv2.seamlessClone (MIXED_CLONE) и возвращает результат.
    При любой ошибке — фолбэк: прямая вставка src в dst по маске.
    blend_strength < 1 линейно смешивает Пуассон-результат с фолбэком.
    """
    try:
        cloned = cv2.seamlessClone(src, dst, mask, (cx, cy), cv2.MIXED_CLONE)

        if blend_strength >= 1.0:
            return cloned

        # Частичное смешивание: Пуассон × strength + fallback × (1-strength)
        fallback = _fallback_paste(src, dst, mask, cx, cy)
        return cv2.addWeighted(cloned, blend_strength, fallback, 1.0 - blend_strength, 0)

    except (cv2.error, Exception):
        return _fallback_paste(src, dst, mask, cx, cy)


def _fallback_paste(
    src: np.ndarray,
    dst: np.ndarray,
    mask: np.ndarray,
    cx: int,
    cy: int,
) -> np.ndarray:
    """Прямая вставка src в dst по бинарной маске (фолбэк без смешивания)."""
    result = dst.copy()
    h, w = src.shape[:2]
    bg_h, bg_w = dst.shape[:2]

    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = x1 + w
    y2 = y1 + h

    # Обрезаем, если выходим за границы
    sx1 = max(0, -x1);  sy1 = max(0, -y1)
    dx1 = max(0, x1);   dy1 = max(0, y1)
    dx2 = min(bg_w, x2); dy2 = min(bg_h, y2)
    sw  = dx2 - dx1;    sh  = dy2 - dy1

    if sw <= 0 or sh <= 0:
        return result

    src_roi  = src [sy1:sy1+sh, sx1:sx1+sw]
    mask_roi = mask[sy1:sy1+sh, sx1:sx1+sw]
    alpha    = (mask_roi[..., None] / 255.0).astype(np.float32)

    result[dy1:dy2, dx1:dx2] = (
        src_roi.astype(np.float32) * alpha +
        result[dy1:dy2, dx1:dx2].astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Устаревший API (оставлен для обратной совместимости)
# ─────────────────────────────────────────────────────────────────────────────

def create_feather_mask(h, w, border_ratio=0.2):
    """Мягкая маска с градиентом у края (старый метод)."""
    y, x = np.ogrid[:h, :w]
    dist_to_edge = np.minimum(
        np.minimum(y, h - 1 - y), np.minimum(x, w - 1 - x)
    ).astype(np.float32)
    max_dist = max(h, w) / 2 * (1.0 - border_ratio)
    return np.clip(dist_to_edge / (max_dist + 1e-6), 0.0, 1.0)
