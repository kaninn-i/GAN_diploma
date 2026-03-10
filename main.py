import os
import torch

from data.extract_crops import extract_crops
from gan.train_gan import train_gan
from augmentation.generate_objects import generate_objects
from augmentation.insert_objects import insert_objects


def run_pipeline(
    dataset_path,
    crops_path="data/crops",
    gan_weights_path="gan_weights",
    generated_path="generated_objects",
    output_dataset="dataset_augmented",
    epochs=30,
    images_per_class=50
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")

    # 1️⃣ Извлечение кропов
    print("\n[1/4] Извлечение кропов...")
    extract_crops(dataset_path, crops_path)

    # 2️⃣ Обучение GAN
    print("\n[2/4] Обучение GAN...")
    train_gan(
        data_path=crops_path,
        save_path=gan_weights_path,
        epochs=epochs,
        device=device
    )

    # Определяем число классов автоматически
    num_classes = len(os.listdir(crops_path))

    # 3️⃣ Генерация объектов
    print("\n[3/4] Генерация синтетических объектов...")
    generate_objects(
        weights_path=os.path.join(gan_weights_path, "generator.pth"),
        output_path=generated_path,
        num_classes=num_classes,
        num_images_per_class=images_per_class,
        device=device
    )

    # 4️⃣ Вставка в датасет
    print("\n[4/4] Вставка объектов в изображения...")
    insert_objects(
        original_dataset=dataset_path,
        generated_objects_path=generated_path,
        output_dataset=output_dataset
    )

    print("\nPipeline завершён успешно.")


if __name__ == "__main__":
    run_pipeline(
        dataset_path="dataset",
        epochs=30,
        images_per_class=50
    )