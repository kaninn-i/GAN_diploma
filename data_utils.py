import os
import random
import cv2
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET


def build_label_index(root_dir, allowed_exts=None):
    """
    Строит словарь { имя_файла_без_расширения: полный_путь_к_txt (или xml) }
    по всем .txt (или xml) файлам в root_dir и подпапках.
    При конфликте имён приоритет у .txt (YOLO), затем .xml (VOC).
    """
    if allowed_exts is None:
        allowed_exts = {'.txt', '.xml'}

    # Сначала собираем все подходящие файлы
    label_index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in allowed_exts:
                continue
            base = os.path.splitext(fname)[0]
            full_path = os.path.join(dirpath, fname)

            if base not in label_index:
                label_index[base] = full_path
            else:
                # При конфликте .txt и .xml — предпочитаем .txt (YOLO)
                existing_ext = os.path.splitext(label_index[base])[1].lower()
                if existing_ext == '.xml' and ext == '.txt':
                    label_index[base] = full_path

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
      class_counts:    {class_id: количество_объектов}
      image_objects:   {путь_к_изображению: (список объектов, путь_к_лейблу)}
      class_name_to_id: {имя_класса: class_id}
                         Для YOLO-датасетов — пустой словарь {}.
                         Для XML (Pascal VOC) — маппинг вида {"car": 0, "person": 1, ...}.
    """
    pairs = find_all_image_label_pairs(root_dir)
    if not pairs:
        raise ValueError(
            "Не найдено ни одной пары изображение-аннотация. "
            "Проверьте, что в датасете есть .jpg/.png и .txt/.xml с одинаковыми именами."
        )

    class_counts = defaultdict(int)
    image_objects = {}
    class_name_to_id = {}   # заполняется только для XML; пуст для YOLO

    for img_path, label_path in pairs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        ext = os.path.splitext(label_path)[1].lower()
        if ext == '.txt':
            objs = parse_yolo_label(label_path, w, h)
        elif ext == '.xml':
            objs = parse_voc_xml_label(label_path, w, h, class_name_to_id)
        else:
            continue

        if objs:
            image_objects[img_path] = (objs, label_path)
            for cls, _, _, _, _ in objs:
                class_counts[cls] += 1

    return dict(class_counts), image_objects, class_name_to_id


def _jitter_padded_window(nx1, ny1, nx2, ny2, w, h, jitter_frac, rng):
    """Случайный сдвиг окна кропа в пределах изображения (та же ширина/высота)."""
    cw = nx2 - nx1
    ch = ny2 - ny1
    if cw < 5 or ch < 5:
        return nx1, ny1, nx2, ny2
    max_dx = max(0, int(cw * jitter_frac))
    max_dy = max(0, int(ch * jitter_frac))
    if max_dx == 0 and max_dy == 0:
        return nx1, ny1, nx2, ny2
    sx = rng.randint(-max_dx, max_dx) if max_dx > 0 else 0
    sy = rng.randint(-max_dy, max_dy) if max_dy > 0 else 0
    sx = max(-nx1, min(sx, w - nx2))
    sy = max(-ny1, min(sy, h - ny2))
    return nx1 + sx, ny1 + sy, nx2 + sx, ny2 + sy


def extract_crops_with_padding(
    image_objects,
    output_dir,
    padding_ratio=0.1,
    crop_size=64,
    jitter_variants=1,
    jitter_frac=0.15,
    seed=42,
):
    """
    Извлекает кропы по YOLO-боксам с паддингом.
    jitter_variants > 1: несколько случайных смещений окна на объект (больше обучающих файлов без новых фото).
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    per_class_count = defaultdict(int)
    jitter_variants = max(1, int(jitter_variants))

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

            for v in range(jitter_variants):
                if v == 0:
                    jx1, jy1, jx2, jy2 = nx1, ny1, nx2, ny2
                else:
                    jx1, jy1, jx2, jy2 = _jitter_padded_window(
                        nx1, ny1, nx2, ny2, w, h, jitter_frac, rng
                    )
                crop = img[jy1:jy2, jx1:jx2]
                if crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue
                crop = cv2.resize(crop, (crop_size, crop_size))
                class_dir = os.path.join(output_dir, f"class_{cls}")
                os.makedirs(class_dir, exist_ok=True)
                suffix = "" if jitter_variants == 1 else f"_v{v}"
                save_name = f"{Path(img_path).stem}_{i}{suffix}.jpg"
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


def validate_dataset(image_objects):
    """
    Проверка качества датасета.

    Возвращает:
        warnings: list[str]
    """
    warnings = []

    for img_path, (objects, _) in image_objects.items():

        img = cv2.imread(img_path)

        if img is None:
            warnings.append(
                f"Битое изображение: {Path(img_path).name}"
            )
            continue

        img_h, img_w = img.shape[:2]

        if len(objects) == 0:
            warnings.append(
                f"Нет объектов: {Path(img_path).name}"
            )

        for i, (cls_id, x1, y1, x2, y2) in enumerate(objects):

            if x1 < 0 or y1 < 0:
                warnings.append(
                    f"{Path(img_path).name}: bbox #{i} имеет отрицательные координаты"
                )

            if x2 > img_w or y2 > img_h:
                warnings.append(
                    f"{Path(img_path).name}: bbox #{i} выходит за границы изображения"
                )

            if x2 <= x1 or y2 <= y1:
                warnings.append(
                    f"{Path(img_path).name}: bbox #{i} имеет некорректный размер"
                )

    return warnings


def parse_voc_xml_label(xml_path, img_w, img_h, class_name_to_id):
    """
    Парсит Pascal VOC XML-аннотацию.
    class_name_to_id — общий словарь {имя: id}, модифицируется на месте.
    Возвращает список (cls_id, x1, y1, x2, y2) в пикселях.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Размеры из XML имеют приоритет над переданными значениями
    size = root.find('size')
    if size is not None:
        w_tag = size.find('width')
        h_tag = size.find('height')
        if w_tag is not None and h_tag is not None:
            xml_w = float(w_tag.text)
            xml_h = float(h_tag.text)
            if xml_w > 0 and xml_h > 0:
                img_w, img_h = xml_w, xml_h

    objects = []
    for obj in root.findall('object'):
        name_tag = obj.find('name')
        if name_tag is None:
            continue
        name = name_tag.text.strip()
        cls_id = class_name_to_id.setdefault(name, len(class_name_to_id))

        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        x1 = max(0, int(xmin))
        y1 = max(0, int(ymin))
        x2 = min(int(img_w), int(xmax))
        y2 = min(int(img_h), int(ymax))

        if x2 > x1 and y2 > y1:
            objects.append((cls_id, x1, y1, x2, y2))

    return objects
