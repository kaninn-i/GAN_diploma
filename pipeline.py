import os
import torch

from dataset_utils import extract_crops
from gan_training import train_gan
from augmentation import generate_objects, insert_objects


def run_pipeline(
    dataset_path,
    crops_path="data/crops",
    gan_weights_path="gan_weights",
    generated_path="generated_objects",
    output_dataset="dataset_augmented",
    epochs=30,
    images_to_generate=50
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")

    # 1
    print("\n[1/4] Извлечение кропов...")
    extract_crops(dataset_path, crops_path)

    # 2
    print("\n[2/4] Обучение GAN...")
    train_gan(
        data_path=crops_path,
        save_path=gan_weights_path,
        epochs=epochs,
        device=device
    )

    # 3
    print("\n[3/4] Генерация объектов...")
    generate_objects(
        weights_path=os.path.join(gan_weights_path, "generator.pth"),
        output_path=generated_path,
        num_images=images_to_generate,
        device=device
    )

    # 4
    print("\n[4/4] Вставка объектов...")
    insert_objects(
        original_dataset=dataset_path,
        generated_objects_path=generated_path,
        output_dataset=output_dataset
    )

    print("\nPipeline завершён.")