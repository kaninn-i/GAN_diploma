import os
import cv2
from pathlib import Path
from collections import defaultdict


def build_label_index(root_dir):
    """
    Строит словарь { имя_файла_без_расширения: полный_путь_к_txt }
    по всем .txt файлам в root_dir и подпапках.
    """
    label_index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.txt'):
                base = os.path.splitext(fname)[0]
                label_index[base] = os.path.join(dirpath, fname)
    return label_index


def find_all_image_label_pairs(root_dir):
    """
    Собирает все изображения (jpg, jpeg, png) из root_dir и всех подпапок,
    ищет для каждого аннотацию по совпадению имени (без расширения) в любом
    месте файловой системы внутри root_dir.
    Возвращает список кортежей (путь_к_изображению, путь_к_аннотации).
    """
    image_exts = {'.jpg', '.jpeg', '.png'}
    label_index = build_label_index(root_dir)

    pairs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in image_exts:
                img_path = os.path.join(dirpath, fname)
                base = os.path.splitext(fname)[0]
                if base in label_index:
                    pairs.append((img_path, label_index[base]))
    return pairs


def parse_yolo_label(label_path, img_w, img_h):
    objects = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            objects.append((cls, x1, y1, x2, y2))
    return objects


def analyze_dataset(root_dir):
    """
    Возвращает:
      class_counts: {class_id: количество_объектов}
      image_objects: {путь_к_изображению: (список объектов, путь_к_лейблу)}
    """
    pairs = find_all_image_label_pairs(root_dir)
    if not pairs:
        raise ValueError("Не найдено ни одной пары изображение-аннотация. "
                         "Проверьте, что в датасете есть .jpg/.png и .txt с одинаковыми именами.")

    class_counts = defaultdict(int)
    image_objects = {}
    for img_path, label_path in pairs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        objs = parse_yolo_label(label_path, w, h)
        if objs:
            image_objects[img_path] = (objs, label_path)
            for cls, _, _, _, _ in objs:
                class_counts[cls] += 1
    return dict(class_counts), image_objects


def extract_crops_with_padding(image_objects, output_dir, padding_ratio=0.1, crop_size=64):
    os.makedirs(output_dir, exist_ok=True)
    per_class_count = defaultdict(int)
    for img_path, (objs, _) in image_objects.items():
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        for i, (cls, x1, y1, x2, y2) in enumerate(objs):
            bw = x2 - x1
            bh = y2 - y1
            pad_w = int(bw * padding_ratio)
            pad_h = int(bh * padding_ratio)
            nx1 = max(0, x1 - pad_w)
            ny1 = max(0, y1 - pad_h)
            nx2 = min(w, x2 + pad_w)
            ny2 = min(h, y2 + pad_h)
            crop = img[ny1:ny2, nx1:nx2]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue
            crop = cv2.resize(crop, (crop_size, crop_size))
            class_dir = os.path.join(output_dir, f"class_{cls}")
            os.makedirs(class_dir, exist_ok=True)
            save_name = f"{Path(img_path).stem}_{i}.jpg"
            cv2.imwrite(os.path.join(class_dir, save_name), crop)
            per_class_count[cls] += 1
    return dict(per_class_count)


def load_dataset_images(image_objects):
    """
    Возвращает список (путь_к_изображению, ширина, высота) для всех изображений с аннотациями.
    """
    image_info = {}
    for img_path in image_objects:
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            image_info[img_path] = (h, w)
    return image_info