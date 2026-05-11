import os
import cv2
import math
import time
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from yolo_validator import validate_yolo

from data_utils import analyze_dataset, extract_crops_with_padding, load_dataset_images
from gan_train import train_gan
from object_generator import generate_objects
from image_integrator import insert_object, match_object_to_background


def split_and_copy(file_pairs, output_root, splits):
    random.shuffle(file_pairs)

    total = len(file_pairs)
    start = 0

    for subset, ratio in splits.items():
        if ratio <= 0:
            continue

        count = min(math.ceil(total * ratio), total - start)
        subset_files = file_pairs[start:start + count]
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


def build_generation_plan(class_counts, balance_to_max=True, override_plan=None):
    if override_plan is not None:
        return override_plan

    if not balance_to_max:
        return {cls: 0 for cls in class_counts}

    target_count = max(class_counts.values())

    return {
        cls: max(0, target_count - count)
        for cls, count in class_counts.items()
    }


def notify_stage(callback, stage_name, stage_num, stage_total):
    if callback:
        callback(stage_name, stage_num, stage_total)


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
    img_size=64,
    blend_strength=0.5,
    progress_callback=None,
    stage_callback=None,
    run_yolo_validation=False,
    yolo_epochs=15
):

    os.makedirs(output_dir, exist_ok=True)

    tmp_root = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)

    stage_timings = {}

    # ======================= Этап 1: Анализ датасета + извлечение кропов ==============================

    notify_stage(stage_callback, "Анализ датасета", 1, 5)

    stage_start = time.time()

    class_counts, image_objects = analyze_dataset(dataset_dir)

    if not class_counts:
        raise ValueError("No valid objects found.")

    crops_dir = os.path.join(tmp_root, "crops")

    extract_crops_with_padding(
        image_objects=image_objects,
        output_dir=crops_dir,
        padding_ratio=padding_ratio,
        crop_size=img_size
    )

    stage_timings["crop_extraction"] = time.time() - stage_start

    # ====================== Этап 2: обучение GAN для каждого класса ===============================

    generation_plan = build_generation_plan(
        class_counts=class_counts,
        balance_to_max=balance_to_max,
        override_plan=generation_plan_override
    )

    notify_stage(stage_callback, "Обучение GAN", 2, 5)

    stage_start = time.time()

    weights_dir = os.path.join(tmp_root, "weights")
    trained_classes = []

    for cls, num_to_generate in generation_plan.items():

        if num_to_generate <= 0:
            continue

        crop_dir = os.path.join(crops_dir, f"class_{cls}")

        if not os.path.exists(crop_dir):
            continue

        if len(os.listdir(crop_dir)) == 0:
            continue

        class_weights_dir = os.path.join(weights_dir, f"class_{cls}")
        os.makedirs(class_weights_dir, exist_ok=True)

        try:
            metrics = train_gan(
                class_dir=crop_dir,
                save_dir=class_weights_dir,
                epochs=epochs,
                device=model_device,
                model_type=model_type,
                img_size=img_size,
                use_ema=True,
                progress_callback=progress_callback
            )

            if metrics is not None:
                trained_classes.append(cls)

        except Exception as e:
            print(f"[TRAIN ERROR] class {cls}: {e}")
            continue

    stage_timings["gan_training"] = time.time() - stage_start

    # ==================== Этап 3: генерация объектов =================================

    notify_stage(stage_callback, "Генерация объектов", 3, 5)

    stage_start = time.time()

    synth_dir = os.path.join(tmp_root, "synth_objects")
    os.makedirs(synth_dir, exist_ok=True)

    synthetic_objects = []

    for cls in trained_classes:

        num_to_generate = generation_plan[cls]

        class_weights_dir = os.path.join(weights_dir, f"class_{cls}")
        class_output_dir = os.path.join(synth_dir, f"class_{cls}")

        ema_path = os.path.join(class_weights_dir, "generator_ema.pth")

        if not os.path.exists(ema_path):
            ema_path = None

        try:
            generate_objects(
                generator_path=os.path.join(class_weights_dir, "generator.pth"),
                output_dir=class_output_dir,
                num_images=num_to_generate,
                latent_dim=128,
                device=model_device,
                model_type=model_type,
                img_size=img_size,
                ema_weights_path=ema_path
            )

        except Exception as e:
            print(f"[GEN ERROR] class {cls}: {e}")
            continue

        for file_name in os.listdir(class_output_dir):
            if file_name.endswith(".png"):
                synthetic_objects.append(
                    (cls, os.path.join(class_output_dir, file_name))
                )

    stage_timings["generation"] = time.time() - stage_start


    # ==================== Этап 4: интеграция объектов в изображения + генерация аннотаций =================================

    notify_stage(stage_callback, "Интеграция объектов", 4, 5)

    stage_start = time.time()

    image_info = load_dataset_images(image_objects)
    bg_images = list(image_info.keys())

    if not bg_images:
        raise RuntimeError("No background images found.")

    random.shuffle(synthetic_objects)

    final_pairs = []

    for img_path, (_, label_path) in image_objects.items():
        final_pairs.append((img_path, label_path))

    temp_generated = []

    object_idx = 0
    image_idx = 0

    pbar = tqdm(total=len(synthetic_objects), desc="Inserting objects")

    while object_idx < len(synthetic_objects):

        bg_path = random.choice(bg_images)

        bg_img = cv2.imread(bg_path)

        if bg_img is None:
            continue

        result_img = bg_img.copy()
        annotations = []

        inserts_count = min(
            random.randint(1, max_objects_per_image),
            len(synthetic_objects) - object_idx
        )

        for _ in range(inserts_count):

            cls, synth_path = synthetic_objects[object_idx]

            synth_img = cv2.imread(synth_path)

            if synth_img is None:
                object_idx += 1
                continue

            try:
                adapted = match_object_to_background(synth_img, bg_img)

                result_img, bbox = insert_object(
                    background=result_img,
                    object_img=adapted,
                    color_adapt=False,
                    blend_strength=blend_strength
                )

            except Exception as e:
                print(f"[INSERT ERROR]: {e}")
                object_idx += 1
                continue

            annotations.append(
                f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
            )

            object_idx += 1
            pbar.update(1)

        img_name = f"aug_{image_idx:06d}.jpg"
        lbl_name = f"aug_{image_idx:06d}.txt"

        img_path = os.path.join(tmp_root, "new_images", img_name)
        lbl_path = os.path.join(tmp_root, "new_images", lbl_name)

        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        cv2.imwrite(img_path, result_img)

        with open(lbl_path, "w") as f:
            f.write("\n".join(annotations))

        temp_generated.append((img_path, lbl_path))

        image_idx += 1

    pbar.close()

    final_pairs.extend(temp_generated)

    stage_timings["integration"] = time.time() - stage_start


    # =================== Этап 5: экспорт датасета ==================================

    notify_stage(stage_callback, "Экспорт датасета", 5, 5)

    stage_start = time.time()

    if split_config:

        total_ratio = sum(split_config.values())

        if abs(total_ratio - 1.0) > 0.01:
            split_config = {
                k: v / total_ratio
                for k, v in split_config.items()
            }

        split_and_copy(final_pairs, output_dir, split_config)

    else:

        img_dir = os.path.join(output_dir, "images")
        lbl_dir = os.path.join(output_dir, "labels")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img_path, label_path in final_pairs:

            base = Path(img_path).stem
            ext = Path(img_path).suffix

            shutil.copy2(
                img_path,
                os.path.join(img_dir, f"{base}{ext}")
            )

            shutil.copy2(
                label_path,
                os.path.join(lbl_dir, f"{base}.txt")
            )

    stage_timings["export"] = time.time() - stage_start

    print(f"Augmented dataset saved to {output_dir}")

# =================== Этап 6: валидация с помощью YOLOv8 ==================

    yolo_results = None

    if run_yolo_validation and split_config:

        notify_stage(stage_callback, "YOLO validation", 6, 6)

        stage_start = time.time()

        try:

            class_names = [f"class_{cls}" for cls in sorted(class_counts.keys())]

            original_yolo_dir = os.path.join(tmp_root, "yolo_original")
            augmented_yolo_dir = os.path.join(tmp_root, "yolo_augmented")

            print("Running baseline YOLO validation...")

            original_metrics = validate_yolo(
                dataset_dir=dataset_dir,
                output_dir=original_yolo_dir,
                class_names=class_names,
                epochs=yolo_epochs
            )

            print("Running augmented YOLO validation...")

            augmented_metrics = validate_yolo(
                dataset_dir=output_dir,
                output_dir=augmented_yolo_dir,
                class_names=class_names,
                epochs=yolo_epochs
            )

            yolo_results = {
                "original": original_metrics,
                "augmented": augmented_metrics,
                "delta_map50": augmented_metrics["map50"] - original_metrics["map50"],
                "delta_recall": augmented_metrics["recall"] - original_metrics["recall"]
            }

        except Exception as e:

            print(f"[YOLO ERROR]: {e}")

            yolo_results = {
                "error": str(e)
            }

        stage_timings["yolo_validation"] = time.time() - stage_start

    return {
        "timings": stage_timings,
        "yolo": yolo_results
    }