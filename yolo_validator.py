import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO


def _find_images(folder):
    """Рекурсивно находит все изображения в папке."""
    exts = {".jpg", ".jpeg", ".png"}
    result = []
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            if Path(f).suffix.lower() in exts:
                result.append(os.path.join(dirpath, f))
    return result


def _is_split_dataset(dataset_dir):
    """Проверяет, есть ли уже разбивка train/val."""
    return (
        os.path.isdir(os.path.join(dataset_dir, "train", "images")) and
        os.path.isdir(os.path.join(dataset_dir, "val", "images"))
    )


def _split_flat_dataset(dataset_dir, output_dir, val_ratio=0.2, seed=42):
    """
    Разбивает плоский датасет (images/ + labels/) на train/val.
    Возвращает путь к папке с разбивкой.
    """
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    # Если нет папки images — ищем изображения прямо в корне
    if not os.path.isdir(images_dir):
        all_images = _find_images(dataset_dir)
        pairs = []
        for img_path in all_images:
            base = Path(img_path).stem
            lbl_candidates = [
                os.path.join(os.path.dirname(img_path), base + ".txt"),
                os.path.join(dataset_dir, "labels", base + ".txt"),
            ]
            for lbl in lbl_candidates:
                if os.path.isfile(lbl):
                    pairs.append((img_path, lbl))
                    break
    else:
        pairs = []
        for img_path in _find_images(images_dir):
            base = Path(img_path).stem
            lbl_path = os.path.join(labels_dir, base + ".txt")
            if os.path.isfile(lbl_path):
                pairs.append((img_path, lbl_path))

    if not pairs:
        raise RuntimeError(
            f"Не найдено ни одной пары изображение+аннотация в {dataset_dir}"
        )

    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for subset, subset_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out = os.path.join(output_dir, subset, "images")
        lbl_out = os.path.join(output_dir, subset, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img_path, lbl_path in subset_pairs:
            shutil.copy2(img_path, os.path.join(img_out, Path(img_path).name))
            shutil.copy2(lbl_path, os.path.join(lbl_out, Path(lbl_path).name))

    return output_dir


def create_dataset_yaml(dataset_dir, yaml_path, class_names=None):
    if class_names is None:
        class_names = ["object"]

    data = {
        "path": os.path.abspath(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "names": class_names,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    return yaml_path


def validate_yolo(
    dataset_dir,
    output_dir,
    class_names,
    model_name="yolov8n.pt",
    epochs=5,
    imgsz=640,
    val_ratio=0.2,
):
    os.makedirs(output_dir, exist_ok=True)

    # Если датасет уже разбит на train/val — используем как есть.
    # Если плоский — разбиваем во временную папку внутри output_dir.
    if _is_split_dataset(dataset_dir):
        ready_dir = dataset_dir
    else:
        ready_dir = os.path.join(output_dir, "split_dataset")
        if not _is_split_dataset(ready_dir):   # кэш: не разбиваем повторно
            _split_flat_dataset(dataset_dir, ready_dir, val_ratio=val_ratio)

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    create_dataset_yaml(
        dataset_dir=ready_dir,
        yaml_path=yaml_path,
        class_names=class_names,
    )

    model = YOLO(model_name)

    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        project=output_dir,
        name="training",
        verbose=False,
        workers=0,
    )

    metrics = model.val(
        data=yaml_path,
        split="val",
        imgsz=imgsz,
        workers=0,
        verbose=False,
    )

    return {
        "map50": float(metrics.box.map50),
        "map5095": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
