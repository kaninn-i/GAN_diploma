"""
stylegan2_ada_trainer.py

Обёртка над официальным stylegan2-ada-pytorch (NVIDIA).
Подготавливает данные, запускает обучение, возвращает путь к чекпоинту.

Установка:
    git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
    pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3

Путь к репо задаётся через переменную окружения STYLEGAN2_ADA_PATH
или автоматически ищется рядом с проектом.
"""

from __future__ import annotations

import os
import re
import sys
import glob
import time
import zipfile
import subprocess
import shutil
from pathlib import Path
from typing import Optional

import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Поиск репозитория
# ─────────────────────────────────────────────────────────────────────────────

_REPO_CANDIDATES = [
    os.environ.get("STYLEGAN2_ADA_PATH", ""),
    "stylegan2-ada-pytorch",
    "../stylegan2-ada-pytorch",
    os.path.expanduser("~/stylegan2-ada-pytorch"),
]


def find_sg2ada_repo() -> str:
    for c in _REPO_CANDIDATES:
        if c and os.path.isfile(os.path.join(c, "train.py")):
            return os.path.abspath(c)
    raise RuntimeError(
        "Репозиторий stylegan2-ada-pytorch не найден.\n"
        "Склонируйте его:\n"
        "  git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git\n"
        "или укажите переменную окружения STYLEGAN2_ADA_PATH."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Авто-параметры по размеру датасета
# ─────────────────────────────────────────────────────────────────────────────

def _auto_cfg(n_images: int, img_size: int) -> str:
    if img_size <= 32 or n_images < 50:
        return "cifar"
    return "auto"


def _auto_kimg(n_images: int) -> int:
    """Количество kimg — тысяч изображений через генератор за обучение."""
    if n_images < 80:
        return 150
    if n_images < 200:
        return 300
    if n_images < 500:
        return 500
    return 800


def _next_power2(x: int) -> int:
    s = 32
    while s < x:
        s *= 2
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Подготовка zip-датасета для SG2-ADA
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset_zip(crop_dir: str, zip_path: str, img_size: int) -> int:
    """
    Упаковывает кропы из crop_dir в zip требуемого SG2-ADA формата.
    Ресайзит всё до img_size × img_size.
    Возвращает количество упакованных изображений.
    """
    exts = {".png", ".jpg", ".jpeg"}
    images = sorted(p for p in Path(crop_dir).iterdir() if p.suffix.lower() in exts)
    if not images:
        raise ValueError(f"Нет изображений в {crop_dir}")

    os.makedirs(Path(zip_path).parent, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if img.shape[0] != img_size or img.shape[1] != img_size:
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".png", img)
            if ok:
                zf.writestr(f"{count:06d}.png", buf.tobytes())
                count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Парсинг прогресса из stdout SG2-ADA
# ─────────────────────────────────────────────────────────────────────────────

# Примеры строк SG2-ADA:
#   tick 1  kimg 1.6  ...  Loss/G/loss 0.5321  Loss/D/loss 1.0213
_KIMG_RE   = re.compile(r"kimg\s+([\d.]+)", re.I)
_GLOSS_RE  = re.compile(r"Loss/G/loss\s+([\d.eE+\-]+)", re.I)
_DLOSS_RE  = re.compile(r"Loss/D/loss\s+([\d.eE+\-]+)", re.I)


def _parse_line(line: str) -> dict:
    info: dict = {}
    m = _KIMG_RE.search(line)
    if m:
        info["kimg"] = float(m.group(1))
    m = _GLOSS_RE.search(line)
    if m:
        info["g_loss"] = float(m.group(1))
    m = _DLOSS_RE.search(line)
    if m:
        info["d_loss"] = float(m.group(1))
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Поиск последнего чекпоинта
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(outdir: str) -> Optional[str]:
    pkls = sorted(glob.glob(os.path.join(outdir, "**", "network-snapshot-*.pkl"), recursive=True))
    if pkls:
        return pkls[-1]
    final = os.path.join(outdir, "network-final.pkl")
    return final if os.path.isfile(final) else None


# ─────────────────────────────────────────────────────────────────────────────
# Главная функция обучения
# ─────────────────────────────────────────────────────────────────────────────

def train_stylegan2_ada(
    class_dir: str,
    save_dir: str,
    img_size: int = 64,
    kimg: int = None,
    cfg: str = None,
    aug: str = "ada",
    mirror: bool = True,
    snap: int = 25,
    device: str = "cuda",
    progress_callback=None,
    sg2ada_repo: str = None,
    resume_pkl: str = None,
    batch_size: int = None,
) -> Optional[dict]:
    """
    Обучает StyleGAN2-ADA на кропах из class_dir.

    Параметры
    ----------
    class_dir         : папка с кропами одного класса (выход extract_crops_with_padding)
    save_dir          : куда сохранять веса, логи и zip датасет
    img_size          : целевое разрешение; округляется до ближайшей степени 2
    kimg              : длина обучения в тысячах сгенерированных изображений (None = авто)
    cfg               : конфиг SG2-ADA ('auto', 'cifar', 'stylegan2'); None = авто
    aug               : режим аугментации ('ada' — рекомендуется для малых датасетов)
    mirror            : горизонтальный flip-аугментация
    snap              : сохранять снапшот каждые N тиков обучения
    device            : 'cuda' или 'cpu'
    progress_callback : (float, str) → None  ← тот же формат, что у train_gan
    sg2ada_repo       : путь к клону репо (None = автопоиск)
    resume_pkl        : .pkl для продолжения / fine-tune
    batch_size        : None = SG2-ADA выбирает сам по cfg

    Возвращает dict с метриками (совместим с тем, что ожидает augment_pipeline),
    или None при ошибке.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Репо
    repo = sg2ada_repo or find_sg2ada_repo()

    # 2. Считаем кропы
    exts = {".png", ".jpg", ".jpeg"}
    n_images = sum(1 for f in Path(class_dir).iterdir() if f.suffix.lower() in exts)
    if n_images < 20:
        print(f"[SG2-ADA] Слишком мало изображений ({n_images} < 20), пропускаю.")
        return None

    # 3. Авто-параметры
    sg2_size = _next_power2(img_size)
    kimg_val = kimg if kimg is not None else _auto_kimg(n_images)
    cfg_val  = cfg  if cfg  is not None else _auto_cfg(n_images, sg2_size)

    print(f"[SG2-ADA] n={n_images} imgs | res={sg2_size}px | cfg={cfg_val} | kimg={kimg_val}")

    # 4. Данные → zip
    zip_path = os.path.join(save_dir, "dataset.zip")
    if progress_callback:
        progress_callback(0.0, f"[SG2-ADA] Упаковка датасета ({n_images} кропов)...")
    n_packed = prepare_dataset_zip(class_dir, zip_path, sg2_size)
    print(f"[SG2-ADA] Упаковано: {n_packed} изображений → {zip_path}")

    # 5. Команда
    outdir = os.path.join(save_dir, "training-runs")
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(repo, "train.py"),
        f"--outdir={outdir}",
        f"--data={zip_path}",
        f"--cfg={cfg_val}",
        f"--aug={aug}",
        f"--kimg={kimg_val}",
        f"--snap={snap}",
        f"--mirror={1 if mirror else 0}",
        "--metrics=none",   # без метрик во время обучения — быстрее
        "--workers=1",
    ]
    if resume_pkl and os.path.isfile(resume_pkl):
        cmd.append(f"--resume={resume_pkl}")
    if batch_size is not None:
        cmd.append(f"--batch={batch_size}")

    print(f"[SG2-ADA] Запуск: {' '.join(cmd)}")

    # 6. Subprocess с парсингом прогресса
    epoch_history: list[dict] = []
    last_kimg  = 0.0
    last_g     = float("inf")
    last_d     = float("inf")
    t0         = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")

    log_path = os.path.join(save_dir, "train.log")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    with open(log_path, "w", encoding="utf-8") as log_f:
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            print(line, end="", flush=True)

            p = _parse_line(line)
            if "kimg" in p:
                last_kimg = p["kimg"]
            if "g_loss" in p:
                last_g = p["g_loss"]
                last_d = p["d_loss"]
                epoch_history.append({
                    "kimg":   round(last_kimg, 1),
                    "g_loss": round(last_g, 5),
                    "d_loss": round(last_d, 5),
                })
            if progress_callback and kimg_val > 0:
                frac = min(1.0, last_kimg / kimg_val)
                progress_callback(
                    frac,
                    f"[SG2-ADA] {last_kimg:.0f}/{kimg_val} kimg"
                    f" | G: {last_g:.4f} | D: {last_d:.4f}",
                )

    proc.wait()

    if proc.returncode != 0:
        print(f"[SG2-ADA] Ошибка обучения (код {proc.returncode}). Лог: {log_path}")
        return None

    # 7. Финальный чекпоинт
    best_pkl = find_latest_checkpoint(outdir)
    if best_pkl is None:
        print("[SG2-ADA] Чекпоинт не найден после обучения.")
        return None

    final_pkl = os.path.join(save_dir, "generator.pkl")
    shutil.copy2(best_pkl, final_pkl)
    print(f"[SG2-ADA] Готово. Чекпоинт → {final_pkl}")

    return {
        # ── Совместимость с полями, которые читает augment_pipeline ──
        "checkpoint":    final_pkl,
        "g_loss":        last_g,
        "d_loss":        last_d,
        "best_g_loss":   last_g,
        "best_epoch":    int(last_kimg),
        "epoch_history": epoch_history,
        "img_size_used": sg2_size,
        # ── Специфика SG2-ADA ────────────────────────────────────────
        "kimg_done":     last_kimg,
        "n_images":      n_packed,
        "elapsed_sec":   round(time.time() - t0, 1),
    }
