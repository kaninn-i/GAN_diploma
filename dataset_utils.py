import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path


# =========================
# Dataset для GAN
# =========================

class CropDataset(Dataset):
    def __init__(self, root_dir, img_size=64):

        self.root_dir = root_dir
        self.img_size = img_size

        self.samples = []
        self.class_to_idx = {}

        entries = os.listdir(root_dir)

        # ============================
        # Проверяем есть ли подпапки
        # ============================

        has_dirs = any(
            os.path.isdir(os.path.join(root_dir, e))
            for e in entries
        )

        # ============================
        # мультикласс режим
        # ============================

        if has_dirs:

            classes = sorted(entries)

            for idx, cls in enumerate(classes):

                cls_path = os.path.join(root_dir, cls)

                if not os.path.isdir(cls_path):
                    continue

                self.class_to_idx[cls] = idx

                for file in os.listdir(cls_path):

                    self.samples.append(
                        (os.path.join(cls_path, file), idx)
                    )

        # ============================
        # один класс
        # ============================

        else:

            self.class_to_idx["class_0"] = 0

            for file in entries:

                self.samples.append(
                    (os.path.join(root_dir, file), 0)
                )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        return img, torch.tensor(label)


# =========================
# YOLO utilities
# =========================

def yolo_to_bbox(label_line, img_w, img_h):
    cls, x, y, w, h = map(float, label_line.split())

    x1 = int((x - w/2) * img_w)
    y1 = int((y - h/2) * img_h)
    x2 = int((x + w/2) * img_w)
    y2 = int((y + h/2) * img_h)

    return int(cls), x1, y1, x2, y2


# =========================
# Подсчет классов из labels
# =========================

def analyze_dataset(dataset_path):

    class_counts = defaultdict(int)

    for root, _, files in os.walk(dataset_path):

        for file in files:

            if not file.endswith(".txt"):
                continue

            # пропускаем README и прочее
            if "README" in file:
                continue

            label_path = os.path.join(root, file)

            with open(label_path) as f:
                lines = f.readlines()

            for line in lines:

                line = line.strip()

                # пустая строка
                if not line:
                    continue

                # комментарий
                if line.startswith("#"):
                    continue

                parts = line.split()

                # не YOLO формат
                if len(parts) < 5:
                    continue

                # первый элемент должен быть классом
                try:
                    cls = int(parts[0])
                except ValueError:
                    continue

                class_counts[cls] += 1

    return dict(class_counts)


# =========================
# Анализ дисбаланса
# =========================

def analyze_imbalance(class_counts):

    if not class_counts:
        return None

    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)

    max_val = class_counts[max_class]
    min_val = class_counts[min_class]

    imbalance = max_val - min_val

    return {
        "max_class": max_class,
        "min_class": min_class,
        "difference": imbalance
    }


# =========================
# Извлечение кропов по классам
# =========================

def extract_crops(dataset_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    image_ext = (".jpg", ".jpeg", ".png")

    # =============================
    # собираем все labels
    # =============================

    label_map = {}

    for root, _, files in os.walk(dataset_path):

        for file in files:

            if not file.endswith(".txt"):
                continue

            if "README" in file:
                continue

            name = os.path.splitext(file)[0]

            label_map[name] = os.path.join(root, file)

    print(f"Найдено labels: {len(label_map)}")

    # =============================
    # собираем изображения
    # =============================

    image_files = []

    for root, _, files in os.walk(dataset_path):

        for file in files:

            if file.lower().endswith(image_ext):
                image_files.append(os.path.join(root, file))

    print(f"Найдено изображений: {len(image_files)}")

    class_counts = defaultdict(int)

    # =============================
    # извлекаем кропы
    # =============================

    for img_path in image_files:

        name = os.path.splitext(os.path.basename(img_path))[0]

        if name not in label_map:
            continue

        label_path = label_map[name]

        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w, _ = img.shape

        with open(label_path) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:
                cls, x1, y1, x2, y2 = yolo_to_bbox(line, w, h)
            except:
                continue

            # clamp bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = img[y1:y2, x1:x2]

            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue

            crop = cv2.resize(crop, (64, 64))

            cls_dir = os.path.join(output_path, f"class_{cls}")
            os.makedirs(cls_dir, exist_ok=True)

            save_path = os.path.join(
                cls_dir,
                f"{name}_{i}.jpg"
            )

            cv2.imwrite(save_path, crop)

            class_counts[cls] += 1

    print("Создано кропов:")

    for cls, count in class_counts.items():
        print(f"class {cls}: {count}")

    return dict(class_counts)