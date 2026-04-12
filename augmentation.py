import os
import torch
from torchvision.utils import save_image
import shutil


def generate_objects(
    weights_path,
    output_path,
    num_images,
    device="cuda",
    batch_size=64,
    model_type="dcgan"
):

    os.makedirs(output_path, exist_ok=True)
    
    if model_type.lower() == "ssd":
        from ssd_model import Generator
    else:
        from gan_model import Generator

    generator = Generator(latent_dim=100).to(device)
    generator.load_state_dict(torch.load(weights_path, map_location=device))
    generator.eval()

    total_images = num_images
    generated = 0
    img_id = 0

    print(f"Генерируем {total_images} изображений...")

    while generated < total_images:

        current_batch = min(batch_size, total_images - generated)

        z = torch.randn(current_batch, 100, 1, 1, device=device)

        with torch.no_grad():
            fake = generator(z)

        fake = (fake + 1) / 2  # [-1,1] → [0,1]

        for i in range(current_batch):

            save_image(
                fake[i],
                os.path.join(output_path, f"generated_{img_id}.png")
            )

            img_id += 1

        generated += current_batch

    print("Генерация завершена.")


def merge_datasets(original_dataset, generated_objects_path, output_dataset):

    images_out = os.path.join(output_dataset, "images")
    labels_out = os.path.join(output_dataset, "labels")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # -------------------------
    # копируем оригинальный dataset
    # -------------------------

    for root, _, files in os.walk(original_dataset):

        for file in files:

            src = os.path.join(root, file)

            if file.lower().endswith((".png", ".jpg", ".jpeg")):

                shutil.copy(src, os.path.join(images_out, file))

            if file.lower().endswith(".txt"):

                shutil.copy(src, os.path.join(labels_out, file))

    print("Оригинальный датасет скопирован")

    # -------------------------
    # добавляем GAN изображения
    # -------------------------

    class_id = 0

    for i, file in enumerate(os.listdir(generated_objects_path)):

        if not file.endswith(".png"):
            continue

        src = os.path.join(generated_objects_path, file)

        new_name = f"gan_{i}.png"

        dst = os.path.join(images_out, new_name)

        shutil.copy(src, dst)

        # создаем YOLO label
        label_path = os.path.join(
            labels_out,
            new_name.replace(".png", ".txt")
        )

        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    print("GAN изображения добавлены")