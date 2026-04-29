import os
import shutil
import random
import math
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from data_utils import (analyze_dataset, extract_crops_with_padding,
                        load_dataset_images)
from gan_train import train_gan
from object_generator import generate_objects
from image_integrator import (insert_object, match_object_to_background)


def split_and_copy(file_list, output_root, splits):
    """
    file_list: список (путь_к_изображению, путь_к_аннотации)
    splits: словарь {'train': 0.7, 'val': 0.2, 'test': 0.1} и т.п.
    Копирует файлы в output_root/train/images, .../labels и т.д.
    """
    random.shuffle(file_list)
    total = len(file_list)
    start = 0
    for subset, ratio in splits.items():
        if ratio <= 0:
            continue
        count = math.ceil(total * ratio) if sum(splits.values()) > 0 else total
        count = min(count, total - start)
        subset_files = file_list[start:start+count]
        start += count

        img_dir = os.path.join(output_root, subset, "images")
        lbl_dir = os.path.join(output_root, subset, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img_path, label_path in subset_files:
            base = Path(img_path).stem
            ext = Path(img_path).suffix
            shutil.copy2(img_path, os.path.join(img_dir, f"{base}{ext}"))
            shutil.copy2(label_path, os.path.join(lbl_dir, f"{base}.txt"))


def run_full_pipeline(
    dataset_dir,
    output_dir,
    epochs=30,
    balance_to_max=True,
    generation_plan_override=None,
    max_objects_per_image=3,
    model_device="cuda",
    padding_ratio=0.1,
    split_config=None,
    model_type="dcgan",
    img_size=64,           # новое
    blend_strength=0.5     # новое (нужно для insert_object)
):
    """
    Параметры:
      generation_plan_override – словарь {class_id: количество} или None.
        Если None, то при balance_to_max=True генерируем разницу до максимума,
        иначе ничего не генерируем (ждём ручного ввода из интерфейса).
      split_config – None (просто images/ и labels/) или словарь с пропорциями.
    """
    os.makedirs(output_dir, exist_ok=True)
    tmp_root = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)

    # Шаг 1: анализ датасета
    class_counts, image_objects = analyze_dataset(dataset_dir)
    if not class_counts:
        raise ValueError("No valid annotations found.")

    # Шаг 2: извлечение кропов
    crops_dir = os.path.join(tmp_root, "crops")
    extract_crops_with_padding(image_objects, crops_dir, padding_ratio, crop_size=img_size)


    # План генерации
    if generation_plan_override is not None:
        generation_plan = generation_plan_override
    else:
        if balance_to_max:
            target_count = max(class_counts.values())
            generation_plan = {cls: max(0, target_count - cnt)
                               for cls, cnt in class_counts.items()}
        else:
            generation_plan = {cls: 0 for cls in class_counts}

     # Шаг 3: обучение GAN и генерация синтетических объектов
    weights_dir = os.path.join(tmp_root, "weights")
    synth_objects_dir = os.path.join(tmp_root, "synth_objects")
    os.makedirs(synth_objects_dir, exist_ok=True)

    synthetic_objects = []  # (class_id, путь_к_png)

    for cls, num_gen in generation_plan.items():
        if num_gen <= 0:
            continue
        class_crop_dir = os.path.join(crops_dir, f"class_{cls}")
        if not os.path.isdir(class_crop_dir) or len(os.listdir(class_crop_dir)) == 0:
            print(f"No crops for class {cls}, skipping.")
            continue
        class_weight_dir = os.path.join(weights_dir, f"class_{cls}")
        os.makedirs(class_weight_dir, exist_ok=True)

        print(f"Training GAN for class {cls}...")
        metrics = train_gan(
            class_crop_dir, class_weight_dir,
            epochs=epochs, device=model_device,
            model_type=model_type, img_size=img_size,
            use_ema=True
        )
        if metrics is None:
            print(f"Training skipped for class {cls} (not enough data). No objects generated.")
            continue

        print(f"Generating {num_gen} objects for class {cls}...")
        gen_output_dir = os.path.join(synth_objects_dir, f"class_{cls}")
        ema_path = os.path.join(class_weight_dir, "generator_ema.pth")
        if not os.path.exists(ema_path):
            ema_path = None

        generate_objects(
            os.path.join(class_weight_dir, "generator.pth"),
            gen_output_dir,
            num_gen,
            latent_dim=128,
            device=model_device,
            model_type=model_type,
            img_size=img_size,
            ema_weights_path=ema_path
        )
        for f in os.listdir(gen_output_dir):
            if f.endswith(".png"):
                synthetic_objects.append((cls, os.path.join(gen_output_dir, f)))

    # Шаг 4: вставка объектов в фоновые изображения
    image_info = load_dataset_images(image_objects)
    bg_image_list = list(image_info.keys())
    if not bg_image_list:
        raise RuntimeError("No background images available.")

    random.shuffle(synthetic_objects)

    # Собираем итоговый список файлов (изображение + аннотация)
    final_pairs = []  # (путь_img, путь_label)

    # Сначала добавляем оригинальные пары
    for img_path, (objs, label_path) in image_objects.items():
        final_pairs.append((img_path, label_path))

    # Генерируем новые изображения
    idx = 0
    obj_idx = 0
    total_objs = len(synthetic_objects)
    pbar = tqdm(total=total_objs, desc="Inserting objects")
    temp_new_images = []

    while obj_idx < total_objs:
        bg_path = random.choice(bg_image_list)
        bg_img = cv2.imread(bg_path)
        bg_h, bg_w = bg_img.shape[:2]
        result_img = bg_img.copy()
        annotations = []
        num_to_insert = min(random.randint(1, max_objects_per_image),
                            total_objs - obj_idx)
        for _ in range(num_to_insert):
            cls, synth_path = synthetic_objects[obj_idx]
            synth_img = cv2.imread(synth_path)
            if synth_img is None:
                obj_idx += 1
                continue
            scale = random.uniform(0.7, 1.5)
            rot = random.uniform(-30, 30)
            obj_h, obj_w = synth_img.shape[:2]
            rad = math.radians(rot)
            nw = int(abs(obj_w * math.cos(rad) * scale) +
                     abs(obj_h * math.sin(rad) * scale)) + 1
            nh = int(abs(obj_h * math.cos(rad) * scale) +
                     abs(obj_w * math.sin(rad) * scale)) + 1
            cx = random.randint(nw//2, max(nw//2+1, bg_w - nw//2))
            cy = random.randint(nh//2, max(nh//2+1, bg_h - nh//2))

            adapted = match_object_to_background(synth_img, bg_img)  # адаптируем ко всему фону
            result_img, bbox_norm = insert_object(
                result_img, adapted,
                mask=None,
                position=(cx, cy),
                scale_range=(0.1, 0.3),
                angle_range=(-30, 30),
                color_adapt=False,
                blend_strength=blend_strength
            )
            annotations.append(f"{cls} {bbox_norm[0]:.6f} {bbox_norm[1]:.6f} "
                               f"{bbox_norm[2]:.6f} {bbox_norm[3]:.6f}")
            obj_idx += 1
            pbar.update(1)

        # Сохраняем временно новое изображение и его аннотацию
        img_filename = f"aug_{idx:06d}.jpg"
        lbl_filename = f"aug_{idx:06d}.txt"
        tmp_img_path = os.path.join(tmp_root, "new_images", img_filename)
        tmp_lbl_path = os.path.join(tmp_root, "new_images", lbl_filename)
        os.makedirs(os.path.dirname(tmp_img_path), exist_ok=True)
        cv2.imwrite(tmp_img_path, result_img)
        with open(tmp_lbl_path, "w") as f:
            f.write("\n".join(annotations))
        temp_new_images.append((tmp_img_path, tmp_lbl_path))
        idx += 1

    pbar.close()
    final_pairs.extend(temp_new_images)

    # Шаг 5: сохранение итогового датасета
    if split_config:
        total_ratio = sum(split_config.values())
        if abs(total_ratio - 1.0) > 0.01:
            split_config = {k: v/total_ratio for k, v in split_config.items()}
        split_and_copy(final_pairs, output_dir, split_config)
    else:
        img_dir = os.path.join(output_dir, "images")
        lbl_dir = os.path.join(output_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for img_path, label_path in final_pairs:
            base = Path(img_path).stem
            ext = Path(img_path).suffix
            shutil.copy2(img_path, os.path.join(img_dir, f"{base}{ext}"))
            shutil.copy2(label_path, os.path.join(lbl_dir, f"{base}.txt"))

    print(f"Augmented dataset saved to {output_dir}")