import os
import cv2
import math
import time
import shutil
import random
from pathlib import Path
from tqdm import tqdm

from data_utils import analyze_dataset, extract_crops_with_padding, load_dataset_images
from gan_train import train_gan
from object_generator import generate_objects
from image_integrator import insert_object, match_object_to_background
from background_clean import remove_labeled_instances_bgr
from gan_metrics import compute_fid_folders
from experiment_utils import create_experiment, save_experiment_config, save_metrics
from yolo_validator import validate_yolo


# Конвертация меток

def _label_to_yolo_txt(label_path, img_path, image_objects):
    """
    Возвращает строки YOLO-формата для заданного label_path.
    .txt — читаем как есть; .xml — конвертируем из image_objects.
    """
    ext = Path(label_path).suffix.lower()

    if ext == ".txt":
        with open(label_path, "r", encoding="utf-8") as f:
            return f.read()

    # XML → YOLO
    if img_path in image_objects:
        objs, _ = image_objects[img_path]
        img = cv2.imread(img_path)
        if img is None:
            return ""
        h, w = img.shape[:2]
        lines = []
        for cls, x1, y1, x2, y2 in objs:
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        return "\n".join(lines)

    return ""


def _write_label(dst_path, label_path, img_path, image_objects):
    """Записывает аннотацию в YOLO .txt формате."""
    content = _label_to_yolo_txt(label_path, img_path, image_objects)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(content)


# Экспорт датасета 

def split_and_copy(file_pairs, output_root, splits, image_objects=None):
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
            ext  = Path(img_path).suffix

            shutil.copy2(img_path, os.path.join(img_dir, f"{base}{ext}"))

            dst_lbl = os.path.join(lbl_dir, f"{base}.txt")
            if image_objects is not None:
                _write_label(dst_lbl, label_path, img_path, image_objects)
            else:
                shutil.copy2(label_path, dst_lbl)


def _export_yolo_dataset(image_objects, output_dir: str) -> str:
    """
    Конвертирует датасет (YOLO txt или Pascal VOC xml) в плоскую YOLO-структуру:
        output_dir/images/*.jpg  +  output_dir/labels/*.txt
    Возвращает output_dir.
    Используется перед передачей XML-датасета в validate_yolo.
    """
    img_out = os.path.join(output_dir, "images")
    lbl_out = os.path.join(output_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img_path, (_, label_path) in image_objects.items():
        base = Path(img_path).stem
        ext  = Path(img_path).suffix
        shutil.copy2(img_path, os.path.join(img_out, f"{base}{ext}"))
        dst_lbl = os.path.join(lbl_out, f"{base}.txt")
        _write_label(dst_lbl, label_path, img_path, image_objects)

    return output_dir


# Generation plan 

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


def _build_class_names(class_counts, class_name_to_id):
    """Список имён классов, упорядоченных по ID."""
    if class_name_to_id:
        id_to_name = {v: k for k, v in class_name_to_id.items()}
        return [id_to_name.get(cls, f"class_{cls}") for cls in sorted(class_counts.keys())]
    return [f"class_{cls}" for cls in sorted(class_counts.keys())]


# Main pipeline 

def run_full_pipeline(
    dataset_dir,
    output_dir,
    epochs=30,
    balance_to_max=True,
    generation_plan_override=None,
    max_objects_per_image=3,
    model_device="cuda",
    padding_ratio=0.1,
    crop_jitter_variants=3,
    crop_jitter_frac=0.15,
    crop_seed=42,
    split_config=None,
    model_type="dcgan",
    img_size=64,
    blend_strength=0.7,
    progress_callback=None,
    stage_callback=None,
    run_yolo_validation=False,
    yolo_epochs=15,
    compute_fid=True,
    gan_train_kwargs=None,
    log_experiment=False,
    use_clean_background=True,
    inpaint_dilate=2,
    inpaint_radius=3,
):
    os.makedirs(output_dir, exist_ok=True)

    tmp_root = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_root, exist_ok=True)

    stage_timings          = {}
    gan_train_kwargs       = gan_train_kwargs or {}
    fid_by_class           = {}
    gan_metrics_by_class   = {}
    epoch_histories_by_cls = {}

    # Анализ датасета + кропы 

    notify_stage(stage_callback, "Анализ датасета", 1, 5)
    stage_start = time.time()

    class_counts, image_objects, class_name_to_id = analyze_dataset(dataset_dir)

    if not class_counts:
        raise ValueError("No valid objects found.")

    crops_dir = os.path.join(tmp_root, "crops")

    extract_crops_with_padding(
        image_objects=image_objects,
        output_dir=crops_dir,
        padding_ratio=padding_ratio,
        crop_size=img_size,
        jitter_variants=crop_jitter_variants,
        jitter_frac=crop_jitter_frac,
        seed=crop_seed,
    )

    stage_timings["crop_extraction"] = time.time() - stage_start

    # Обучение GAN 

    generation_plan = build_generation_plan(
        class_counts=class_counts,
        balance_to_max=balance_to_max,
        override_plan=generation_plan_override,
    )

    notify_stage(stage_callback, "Обучение GAN", 2, 5)
    stage_start = time.time()

    weights_dir     = os.path.join(tmp_root, "weights")
    trained_classes = []

    for cls, num_to_generate in generation_plan.items():
        if num_to_generate <= 0:
            continue

        crop_dir = os.path.join(crops_dir, f"class_{cls}")
        if not os.path.exists(crop_dir) or len(os.listdir(crop_dir)) == 0:
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
                progress_callback=progress_callback,
                **gan_train_kwargs,
            )

            if metrics is not None:
                trained_classes.append(cls)
                gan_metrics_by_class[int(cls)]   = metrics
                epoch_histories_by_cls[int(cls)] = metrics.get("epoch_history", [])

                actual_size = metrics.get("img_size_used", img_size)
                if actual_size != img_size:
                    print(f"[INFO] class_{cls}: img_size adjusted to {actual_size}")

        except Exception as e:
            print(f"[TRAIN ERROR] class {cls}: {e}")
            continue

    stage_timings["gan_training"] = time.time() - stage_start

    # Генерация объектов 

    notify_stage(stage_callback, "Генерация объектов", 3, 5)
    stage_start = time.time()

    synth_dir = os.path.join(tmp_root, "synth_objects")
    os.makedirs(synth_dir, exist_ok=True)

    synthetic_objects = []

    for cls in trained_classes:
        num_to_generate = generation_plan[cls]

        class_weights_dir = os.path.join(weights_dir, f"class_{cls}")
        class_output_dir  = os.path.join(synth_dir, f"class_{cls}")

        actual_size = gan_metrics_by_class.get(int(cls), {}).get("img_size_used", img_size)

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
                img_size=actual_size,
                ema_weights_path=ema_path,
            )
        except Exception as e:
            print(f"[GEN ERROR] class {cls}: {e}")
            continue

        if compute_fid:
            crop_dir_cls = os.path.join(crops_dir, f"class_{cls}")
            if os.path.isdir(crop_dir_cls) and os.path.isdir(class_output_dir):
                fid_val = compute_fid_folders(
                    crop_dir_cls,
                    class_output_dir,
                    device=model_device,
                )
                if fid_val is not None:
                    fid_by_class[int(cls)] = fid_val
                    print(f"[FID] class_{cls}: {fid_val:.2f}")

        for file_name in os.listdir(class_output_dir):
            if file_name.endswith(".png"):
                synthetic_objects.append(
                    (cls, os.path.join(class_output_dir, file_name))
                )

    stage_timings["generation"] = time.time() - stage_start

    # Интеграция объектов 

    notify_stage(stage_callback, "Интеграция объектов", 4, 5)
    stage_start = time.time()

    image_info = load_dataset_images(image_objects)
    bg_images  = list(image_info.keys())

    if not bg_images:
        raise RuntimeError("No background images found.")

    random.shuffle(synthetic_objects)

    original_pairs = [
        (img_path, label_path)
        for img_path, (_, label_path) in image_objects.items()
    ]

    temp_generated = []
    object_idx = 0
    image_idx  = 0

    pbar = tqdm(total=len(synthetic_objects), desc="Inserting objects")

    while object_idx < len(synthetic_objects):
        bg_path = random.choice(bg_images)
        bg_img  = cv2.imread(bg_path)

        if bg_img is None:
            continue

        if use_clean_background and bg_path in image_objects:
            objs_on_bg, _ = image_objects[bg_path]
            result_img = remove_labeled_instances_bgr(
                bg_img,
                objs_on_bg,
                dilate=int(inpaint_dilate),
                inpaint_radius=int(inpaint_radius),
            )
        else:
            result_img = bg_img.copy()

        annotations = []

        inserts_count = min(
            random.randint(1, max_objects_per_image),
            len(synthetic_objects) - object_idx,
        )

        for _ in range(inserts_count):
            cls, synth_path = synthetic_objects[object_idx]
            synth_img = cv2.imread(synth_path)

            if synth_img is None:
                object_idx += 1
                continue

            try:
                adapted = match_object_to_background(synth_img, result_img)
                result_img, bbox = insert_object(
                    background=result_img,
                    object_img=adapted,
                    color_adapt=False,
                    blend_strength=blend_strength,
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

        aug_img_path = os.path.join(tmp_root, "new_images", img_name)
        aug_lbl_path = os.path.join(tmp_root, "new_images", lbl_name)
        os.makedirs(os.path.dirname(aug_img_path), exist_ok=True)

        cv2.imwrite(aug_img_path, result_img)
        with open(aug_lbl_path, "w") as f:
            f.write("\n".join(annotations))

        temp_generated.append((aug_img_path, aug_lbl_path))
        image_idx += 1

    pbar.close()

    final_pairs = original_pairs + temp_generated
    stage_timings["integration"] = time.time() - stage_start

    # Экспорт 

    notify_stage(stage_callback, "Экспорт датасета", 5, 5)
    stage_start = time.time()

    if split_config:
        total_ratio = sum(split_config.values())
        if abs(total_ratio - 1.0) > 0.01:
            split_config = {k: v / total_ratio for k, v in split_config.items()}
        split_and_copy(final_pairs, output_dir, split_config, image_objects=image_objects)
    else:
        img_dir = os.path.join(output_dir, "images")
        lbl_dir = os.path.join(output_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for img_path, label_path in final_pairs:
            base = Path(img_path).stem
            ext  = Path(img_path).suffix
            shutil.copy2(img_path, os.path.join(img_dir, f"{base}{ext}"))
            dst_lbl = os.path.join(lbl_dir, f"{base}.txt")
            _write_label(dst_lbl, label_path, img_path, image_objects)

    stage_timings["export"] = time.time() - stage_start
    print(f"Augmented dataset saved to {output_dir}")

    # YOLO validation 

    yolo_results = None

    if run_yolo_validation and split_config:
        notify_stage(stage_callback, "Валидация YOLO", 6, 6)
        stage_start = time.time()

        try:
            class_names = _build_class_names(class_counts, class_name_to_id)

            is_xml_dataset = bool(class_name_to_id)

            if is_xml_dataset:
                orig_yolo_converted = os.path.join(tmp_root, "yolo_orig_converted")
                _export_yolo_dataset(image_objects, orig_yolo_converted)
                original_yolo_dir_src = orig_yolo_converted
            else:
                original_yolo_dir_src = dataset_dir

            original_yolo_out  = os.path.join(tmp_root, "yolo_original")
            augmented_yolo_out = os.path.join(tmp_root, "yolo_augmented")

            print("Running baseline YOLO validation...")
            original_metrics = validate_yolo(
                dataset_dir=original_yolo_dir_src,
                output_dir=original_yolo_out,
                class_names=class_names,
                epochs=yolo_epochs,
            )

            print("Running augmented YOLO validation...")
            augmented_metrics = validate_yolo(
                dataset_dir=output_dir,
                output_dir=augmented_yolo_out,
                class_names=class_names,
                epochs=yolo_epochs,
            )

            yolo_results = {
                "original":    original_metrics,
                "augmented":   augmented_metrics,
                "delta_map50": augmented_metrics["map50"] - original_metrics["map50"],
                "delta_recall": augmented_metrics["recall"] - original_metrics["recall"],
            }

        except Exception as e:
            print(f"[YOLO ERROR]: {e}")
            yolo_results = {"error": str(e)}

        stage_timings["yolo_validation"] = time.time() - stage_start

    if log_experiment:
        exp_dir = create_experiment()
        save_experiment_config(
            exp_dir,
            {
                "dataset_dir":          dataset_dir,
                "output_dir":           output_dir,
                "model_type":           model_type,
                "epochs":               epochs,
                "img_size":             img_size,
                "crop_jitter_variants": crop_jitter_variants,
                "crop_jitter_frac":     crop_jitter_frac,
                "gan_train_kwargs":     gan_train_kwargs,
                "compute_fid":          compute_fid,
                "run_yolo_validation":  run_yolo_validation,
                "class_name_to_id":     class_name_to_id,
            },
        )
        save_metrics(
            exp_dir,
            {
                "timings":     stage_timings,
                "gan_fid":     fid_by_class,
                "gan_metrics": gan_metrics_by_class,
                "yolo":        yolo_results,
            },
        )

    return {
        "timings":          stage_timings,
        "yolo":             yolo_results,
        "gan_fid":          fid_by_class or None,
        "gan_metrics":      gan_metrics_by_class or None,
        "epoch_histories":  epoch_histories_by_cls,
        "class_name_to_id": class_name_to_id,
        "synth_dir":        synth_dir,
    }
