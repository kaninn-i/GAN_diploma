"""
Интеграция объектов в фоновые изображения.

Метод: локальная цветовая адаптация к патчу фона в точке вставки +
       мягкая эллиптическая alpha-маска с cosine-спадом (feather).
       Без Пуассонова смешивания — нет артефактов «мазни».
"""

import cv2
import numpy as np
import random


# ─────────────────────────────────────────────────────────────────────────────
# Цветовая адаптация
# ─────────────────────────────────────────────────────────────────────────────

def _local_patch_stats(
    background: np.ndarray, cx: int, cy: int, radius: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Среднее и стд. отклонение локального патча фона вокруг точки (cx, cy)."""
    h, w = background.shape[:2]
    x1, x2 = max(0, cx - radius), min(w, cx + radius)
    y1, y2 = max(0, cy - radius), min(h, cy + radius)
    patch = background[y1:y2, x1:x2]
    if patch.size < 9:
        patch = background
    mean, std = cv2.meanStdDev(patch)
    return np.array(mean).flatten(), np.array(std).flatten()


def adjust_brightness_contrast(
    obj_bgr: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> np.ndarray:
    obj_mean, obj_std = cv2.meanStdDev(obj_bgr)
    obj_mean = np.array(obj_mean).flatten()
    obj_std  = np.array(obj_std).flatten()
    scale = np.divide(
        target_std, obj_std, out=np.ones_like(target_std), where=obj_std != 0
    )
    scale = np.clip(scale, 0.5, 2.0)          # не перекрашивать радикально
    adjusted = (obj_bgr.astype(np.float32) - obj_mean) * scale + target_mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def match_object_to_background(obj_img: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Глобальная адаптация (используется в pipeline до вставки)."""
    bg_mean, bg_std = cv2.meanStdDev(background)
    return adjust_brightness_contrast(
        obj_img, np.array(bg_mean).flatten(), np.array(bg_std).flatten()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Маска
# ─────────────────────────────────────────────────────────────────────────────

def _feather_mask(h: int, w: int, border_frac: float = 0.22) -> np.ndarray:
    """
    Float-маска [0..1]: 1.0 в центре, 0.0 у края.
    Cosine-спад в зоне шириной border_frac относительно радиуса.
    """
    y_idx = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x_idx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(y_idx, x_idx, indexing="ij")
    dist   = np.sqrt(np.clip(xx**2 + yy**2, 0.0, None))

    inner  = 1.0 - border_frac
    mask   = np.where(
        dist <= inner,
        1.0,
        np.where(
            dist <= 1.0,
            0.5 * (1.0 + np.cos(np.pi * (dist - inner) / (border_frac + 1e-6))),
            0.0,
        ),
    )
    return mask.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция вставки
# ─────────────────────────────────────────────────────────────────────────────

def insert_object(
    background: np.ndarray,
    object_img: np.ndarray,
    mask: np.ndarray | None = None,
    position: tuple | None = None,
    scale_range: tuple = (0.10, 0.30),
    angle_range: tuple = (-15, 15),
    color_adapt: bool = True,
    blend_strength: float = 0.85,
) -> tuple[np.ndarray, tuple]:
    """
    Вставляет object_img в background.

    Шаги:
      1. Вычисление целевого размера объекта.
      2. Выбор/проверка центра вставки.
      3. Локальная цветовая адаптация к патчу фона.
      4. Ресайз + вращение объекта.
      5. Alpha-blend с feather-маской.

    Возвращает (result_bgr, (x_center, y_center, bw, bh)) — нормированные координаты.
    """
    bg_h, bg_w = background.shape[:2]
    obj_h, obj_w = object_img.shape[:2]

    # 1. Целевой размер
    min_side  = min(bg_h, bg_w)
    raw_scale = random.uniform(*scale_range) * min_side / max(obj_h, obj_w)
    new_w     = max(8, int(obj_w * raw_scale))
    new_h     = max(8, int(obj_h * raw_scale))
    angle     = random.uniform(*angle_range)

    # 2. Центр вставки (объект полностью внутри изображения)
    half_w, half_h = new_w // 2, new_h // 2
    margin_x = half_w + 1
    margin_y = half_h + 1

    if position is None:
        cx = random.randint(margin_x, max(margin_x, bg_w - margin_x))
        cy = random.randint(margin_y, max(margin_y, bg_h - margin_y))
    else:
        cx = int(np.clip(position[0], margin_x, bg_w - margin_x))
        cy = int(np.clip(position[1], margin_y, bg_h - margin_y))

    # 3. Локальная цветовая адаптация
    if color_adapt:
        patch_radius        = max(20, int(max(new_w, new_h) * 0.6))
        local_mean, local_std = _local_patch_stats(background, cx, cy, radius=patch_radius)
        object_img          = adjust_brightness_contrast(object_img, local_mean, local_std)

    # 4. Ресайз + вращение
    obj_ready = cv2.resize(object_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if angle != 0.0:
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
        obj_ready = cv2.warpAffine(
            obj_ready, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    # 5. Маска
    if mask is None:
        alpha = _feather_mask(new_h, new_w, border_frac=0.22)
    else:
        alpha = cv2.resize(
            mask.astype(np.float32) / 255.0, (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )
        alpha = alpha * _feather_mask(new_h, new_w, border_frac=0.15)

    alpha = alpha * float(blend_strength)

    # 6. Alpha-blend
    result = background.copy()

    x1 = cx - half_w;  y1 = cy - half_h
    x2 = x1 + new_w;   y2 = y1 + new_h

    sx1 = max(0, -x1);  sy1 = max(0, -y1)
    dx1 = max(0, x1);   dy1 = max(0, y1)
    dx2 = min(bg_w, x2); dy2 = min(bg_h, y2)
    sw  = dx2 - dx1;    sh  = dy2 - dy1

    if sw > 0 and sh > 0:
        src_roi = obj_ready[sy1:sy1+sh, sx1:sx1+sw].astype(np.float32)
        dst_roi = result   [dy1:dy2,    dx1:dx2   ].astype(np.float32)
        a       = alpha    [sy1:sy1+sh, sx1:sx1+sw][..., None]
        result[dy1:dy2, dx1:dx2] = np.clip(
            src_roi * a + dst_roi * (1.0 - a), 0, 255
        ).astype(np.uint8)

    # 7. YOLO bbox (нормированные координаты)
    cx_n = ((dx1 + dx2) / 2) / bg_w
    cy_n = ((dy1 + dy2) / 2) / bg_h
    bw_n = (dx2 - dx1) / bg_w
    bh_n = (dy2 - dy1) / bg_h

    return result, (cx_n, cy_n, bw_n, bh_n)


# Псевдоним для обратной совместимости
def create_feather_mask(h, w, border_ratio=0.2):
    return _feather_mask(h, w, border_frac=border_ratio)