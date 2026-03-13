import os
import torch
import random
import cv2
from torchvision.utils import save_image

from gan_model import Generator


def generate_objects(
    weights_path,
    output_path,
    num_images,
    device="cuda",
    batch_size=64
):

    os.makedirs(output_path, exist_ok=True)

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



def insert_objects(original_dataset, generated_objects_path, output_dataset):

    os.makedirs(output_dataset, exist_ok=True)

    generated_images = [
        f for f in os.listdir(generated_objects_path)
        if f.endswith(".png")
    ]

    original_images = [
        f for f in os.listdir(original_dataset)
        if f.endswith(".png") or f.endswith(".jpg")
    ]

    print(f"Найдено {len(generated_images)} сгенерированных объектов")

    for img_name in original_images:

        img_path = os.path.join(original_dataset, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        h, w = image.shape[:2]

        # выбираем случайный объект
        obj_name = random.choice(generated_images)
        obj_path = os.path.join(generated_objects_path, obj_name)

        obj = cv2.imread(obj_path)

        if obj is None:
            continue

        oh, ow = obj.shape[:2]

        # случайная позиция
        x = random.randint(0, max(1, w - ow))
        y = random.randint(0, max(1, h - oh))

        # вставка
        image[y:y+oh, x:x+ow] = obj

        save_path = os.path.join(output_dataset, img_name)
        cv2.imwrite(save_path, image)

    print("Вставка объектов завершена.")