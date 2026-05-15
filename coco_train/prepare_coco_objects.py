# prepare_coco_objects.py
import os
import sys
import time
import json
import shutil
import urllib.request
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np
from tqdm import tqdm

# Конфигурация
COCO_DIR = "coco_data"                # куда скачивать
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"   # ~18G
VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip" # ~1G
OBJECTS_OUT = "coco_objects_128"      # сюда сохраняем кропы
CROP_SIZE = 128
PADDING_RATIO = 0.15

def download_file(url, dest):
    """Скачивание с возобновлением."""
    if os.path.exists(dest):
        print(f"Файл {dest} уже существует, пропускаем загрузку.")
        return
    print(f"Скачиваю {url} ...")
    urllib.request.urlretrieve(url, dest)
    print("Готово.")

def extract_zip(src, dst_dir):
    print(f"Распаковываю {src} ...")
    with ZipFile(src, 'r') as z:
        z.extractall(dst_dir)
    print("Распаковано.")

def load_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def extract_crops(annotations, images_dir, output_dir, padding_ratio=0.15, crop_size=128):
    os.makedirs(output_dir, exist_ok=True)
    images_info = {img['id']: img['file_name'] for img in annotations['images']}

    # Собираем все аннотации (объекты)
    boxes = []
    for ann in annotations['annotations']:
        if ann.get('bbox') is None:
            continue
        bbox = ann['bbox']  # [x, y, width, height]
        image_id = ann['image_id']
        if image_id not in images_info:
            continue
        boxes.append((image_id, bbox))

    print(f"Всего объектов: {len(boxes)}")
    for image_id, bbox in tqdm(boxes):
        img_path = os.path.join(images_dir, images_info[image_id])
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        x, y, bw, bh = map(int, bbox)

        # Расширение рамки
        pad_w = int(bw * padding_ratio)
        pad_h = int(bh * padding_ratio)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w)
        y2 = min(h, y + bh + pad_h)

        crop = img[y1:y2, x1:x2]
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        crop = cv2.resize(crop, (crop_size, crop_size))

        out_name = f"coco_{image_id}_{x}_{y}.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), crop)

def main():
    # 1. Скачиваем аннотации
    os.makedirs(COCO_DIR, exist_ok=True)
    ann_zip = os.path.join(COCO_DIR, "annotations_trainval2017.zip")
    download_file(ANNOTATIONS_URL, ann_zip)
    extract_zip(ann_zip, COCO_DIR)

    # 2. Скачиваем изображения train и val
    train_zip = os.path.join(COCO_DIR, "train2017.zip")
    download_file(IMAGES_URL, train_zip)
    extract_zip(train_zip, COCO_DIR)  # появится папка train2017

    val_zip = os.path.join(COCO_DIR, "val2017.zip")
    download_file(VAL_IMAGES_URL, val_zip)
    extract_zip(val_zip, COCO_DIR)    # появится папка val2017

    # 3. Обработка train
    train_ann_file = os.path.join(COCO_DIR, "annotations", "instances_train2017.json")
    train_imgs_dir = os.path.join(COCO_DIR, "train2017")
    print("Обрабатываю train2017...")
    ann_train = load_annotations(train_ann_file)
    extract_crops(ann_train, train_imgs_dir, os.path.join(OBJECTS_OUT, "train"),
                  padding_ratio=PADDING_RATIO, crop_size=CROP_SIZE)

    # 4. Обработка val
    val_ann_file = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")
    val_imgs_dir = os.path.join(COCO_DIR, "val2017")
    print("Обрабатываю val2017...")
    ann_val = load_annotations(val_ann_file)
    extract_crops(ann_val, val_imgs_dir, os.path.join(OBJECTS_OUT, "val"),
                  padding_ratio=PADDING_RATIO, crop_size=CROP_SIZE)

    print(f"Готово! Объекты сохранены в {OBJECTS_OUT}")

if __name__ == "__main__":
    main()