import os
import torch
import shutil
from pathlib import Path
from torchvision.utils import save_image



# ===============================
# Генерация объектов GAN
# ===============================

def generate_objects(
    weights_path,
    output_path,
    num_images,
    device="cuda",
    batch_size=64,
    model_type="dcgan"
):

    os.makedirs(output_path, exist_ok=True)

    # выбор модели
    if model_type.lower() == "ssd":
        from ssd_model import Generator
    else:
        from gan_model import Generator

    generator = Generator(latent_dim=100).to(device)
    generator.load_state_dict(torch.load(weights_path, map_location=device))
    generator.eval()

    generated = 0
    img_id = 0

    while generated < num_images:

        current_batch = min(batch_size, num_images - generated)

        z = torch.randn(current_batch, 100, 1, 1, device=device)

        with torch.no_grad():
            fake = generator(z)

        fake = (fake + 1) / 2

        for i in range(current_batch):

            save_image(
                fake[i],
                os.path.join(output_path, f"generated_{img_id}.png")
            )

            img_id += 1

        generated += current_batch

    return img_id


# ===============================
# Копирование оригинального dataset
# ===============================

def copy_original_dataset(original_dataset, output_dataset):

    images_dst = os.path.join(output_dataset, "images")
    labels_dst = os.path.join(output_dataset, "labels")

    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)

    image_ext = (".jpg", ".jpeg", ".png")

    img_id = 0

    for root, _, files in os.walk(original_dataset):

        for file in files:

            src = os.path.join(root, file)

            # копируем изображения
            if file.lower().endswith(image_ext):

                new_name = f"orig_{img_id}{Path(file).suffix}"

                shutil.copy(
                    src,
                    os.path.join(images_dst, new_name)
                )

                # ищем label рядом
                label_src = os.path.splitext(src)[0] + ".txt"

                if os.path.exists(label_src):

                    shutil.copy(
                        label_src,
                        os.path.join(
                            labels_dst,
                            new_name.replace(Path(file).suffix, ".txt")
                        )
                    )

                img_id += 1


# ===============================
# merge datasets (мультикласс)
# ===============================

def merge_datasets(
    original_dataset,
    generated_objects_path,
    output_dataset
):
    
    if not os.path.exists(generated_objects_path):
        print("Нет сгенерированных изображений — копируем только оригинальный датасет")
        copy_original_dataset(original_dataset, output_dataset)
        return []

    copy_original_dataset(original_dataset, output_dataset)

    images_out = os.path.join(output_dataset, "images")
    labels_out = os.path.join(output_dataset, "labels")

    img_id = 0
    metrics = []

    # структура:
    # generated/
    #   class_0/
    #   class_1/

    for class_dir in sorted(os.listdir(generated_objects_path)):

        class_path = os.path.join(generated_objects_path, class_dir)

        if not os.path.isdir(class_path):
            continue

        # class_0 -> 0
        class_id = int(class_dir.split("_")[1])

        class_count = 0

        for file in os.listdir(class_path):

            if not file.endswith(".png"):
                continue

            src = os.path.join(class_path, file)

            new_name = f"gan_{class_id}_{img_id}.png"
            dst = os.path.join(images_out, new_name)

            shutil.copy(src, dst)

            # создаем YOLO label
            label_path = os.path.join(
                labels_out,
                new_name.replace(".png", ".txt")
            )

            with open(label_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

            img_id += 1
            class_count += 1

        metrics.append({
            "class": class_id,
            "generated": class_count
        })

    return metrics