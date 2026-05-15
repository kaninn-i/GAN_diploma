"""
Fréchet Inception Distance между двумя папками изображений (Inception v3 pool).
Без внешних зависимостей кроме torch/torchvision/numpy.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import inception_v3, Inception_V3_Weights


def _symmetric_matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    """PSD matrix square root для ковариаций."""
    mat = (mat + mat.T) * 0.5
    w, v = np.linalg.eigh(mat)
    w = np.maximum(w, 1e-6)
    return (v * np.sqrt(w)) @ v.T


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean = _symmetric_matrix_sqrt(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _symmetric_matrix_sqrt((sigma1 + offset) @ (sigma2 + offset))
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


class _InceptionFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        net.aux_logits = False
        net.fc = nn.Identity()
        self.net = net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)


def _iter_images(folder, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    folder = Path(folder)
    if not folder.is_dir():
        return
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in extensions:
            yield str(p)


def _collect_activations(image_paths, model, device, batch_size, transform, max_images):
    acts = []
    batch_imgs = []
    n = 0
    for path in image_paths:
        if n >= max_images:
            break
        try:
            img = Image.open(path).convert("RGB")
        except OSError:
            continue
        batch_imgs.append(transform(img))
        n += 1
        if len(batch_imgs) >= batch_size:
            x = torch.stack(batch_imgs, dim=0).to(device)
            feat = model(x).cpu().numpy()
            acts.append(feat)
            batch_imgs = []
    if batch_imgs:
        x = torch.stack(batch_imgs, dim=0).to(device)
        feat = model(x).cpu().numpy()
        acts.append(feat)
    if not acts:
        return None
    return np.concatenate(acts, axis=0)


def compute_fid_folders(
    real_dir,
    fake_dir,
    device="cuda",
    batch_size=32,
    max_images_per_split=400,
    dims=2048,
):
    """
    FID между реальными (jpg) и сгенерированными (png/jpg) изображениями.
    Возвращает float или None при недостатке данных / ошибке.
    """
    real_paths = list(_iter_images(real_dir))
    fake_paths = list(_iter_images(fake_dir))
    if len(real_paths) < 2 or len(fake_paths) < 2:
        return None

    real_paths = real_paths[:max_images_per_split]
    fake_paths = fake_paths[:max_images_per_split]

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = _InceptionFeatureExtractor(dev)
    transform = T.Compose(
        [
            T.Resize((299, 299), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    act_r = _collect_activations(real_paths, model, dev, batch_size, transform, max_images_per_split)
    act_f = _collect_activations(fake_paths, model, dev, batch_size, transform, max_images_per_split)
    if act_r is None or act_f is None:
        return None
    if act_r.shape[1] != dims or act_f.shape[1] != dims:
        return None

    mu_r = np.mean(act_r, axis=0)
    sigma_r = np.cov(act_r, rowvar=False)
    mu_f = np.mean(act_f, axis=0)
    sigma_f = np.cov(act_f, rowvar=False)
    fid = _frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
    return float(max(fid, 0.0))
