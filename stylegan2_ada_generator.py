"""
stylegan2_ada_generator.py

Генерация изображений из обученного StyleGAN2-ADA чекпоинта (.pkl).
Совместим с интерфейсом generate_objects из object_generator.py.
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка генератора из .pkl
# ─────────────────────────────────────────────────────────────────────────────

def load_sg2ada_generator(pkl_path: str, device: str, sg2ada_repo: str = None):
    """
    Загружает G_ema из чекпоинта SG2-ADA.

    Репо должно быть в PYTHONPATH, иначе pickle не найдёт dnnlib/legacy.
    sg2ada_repo задаётся явно или через STYLEGAN2_ADA_PATH.
    """
    repo = _find_repo(sg2ada_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    # Нужен legacy для загрузки pkl
    try:
        import legacy  # noqa: F401  — часть stylegan2-ada-pytorch
    except ImportError:
        raise ImportError(
            f"Не удалось импортировать legacy из репо SG2-ADA.\n"
            f"Убедитесь, что {repo} содержит legacy.py и dnnlib/."
        )

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # В чекпоинте может быть G или G_ema; G_ema предпочтительнее
    G = data.get("G_ema", data.get("G", None))
    if G is None:
        raise ValueError(f"Чекпоинт {pkl_path} не содержит G или G_ema.")

    G = G.eval().to(device)
    for p in G.parameters():
        p.requires_grad_(False)
    return G


def _find_repo(sg2ada_repo: Optional[str]) -> str:
    from stylegan2_ada_trainer import find_sg2ada_repo
    return sg2ada_repo or find_sg2ada_repo()


# ─────────────────────────────────────────────────────────────────────────────
# Генерация батча
# ─────────────────────────────────────────────────────────────────────────────

def _generate_batch(G, batch_size: int, device: str, truncation_psi: float = 0.7) -> list[np.ndarray]:
    """
    Возвращает список BGR uint8 изображений размером [H, W, 3].
    """
    z = torch.randn([batch_size, G.z_dim], device=device)
    c = None   # без условия на класс

    with torch.no_grad():
        imgs = G(z, c, truncation_psi=truncation_psi, noise_mode="const")

    # SG2-ADA выдаёт [-1, 1] → [0, 255]
    imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # NCHW → NHWC, RGB

    return [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция генерации (совместима с object_generator.generate_objects)
# ─────────────────────────────────────────────────────────────────────────────

def generate_objects_sg2ada(
    generator_path: str,
    output_dir: str,
    num_images: int,
    device: str = "cuda",
    batch_size: int = 16,
    truncation_psi: float = 0.7,
    sg2ada_repo: str = None,
    progress_callback=None,
    **_kwargs,            # проглатывает latent_dim, img_size и др. от GAN-интерфейса
) -> int:
    """
    Генерирует num_images изображений из чекпоинта generator_path (.pkl).

    truncation_psi:
        0.5 – более усреднённые (safe) изображения
        0.7 – баланс качества и разнообразия (рекомендуется)
        1.0 – максимальное разнообразие, возможны артефакты
    """
    os.makedirs(output_dir, exist_ok=True)

    G = load_sg2ada_generator(generator_path, device, sg2ada_repo)

    saved    = 0
    img_id   = 0
    bs       = min(batch_size, num_images)

    while saved < num_images:
        current = min(bs, num_images - saved)
        imgs    = _generate_batch(G, current, device, truncation_psi)

        for bgr in imgs:
            out_path = os.path.join(output_dir, f"synth_{img_id:06d}.png")
            cv2.imwrite(out_path, bgr)
            img_id += 1
            saved  += 1

        if progress_callback:
            progress_callback(
                saved / num_images,
                f"[SG2-ADA] Генерация: {saved}/{num_images}",
            )

    return saved
