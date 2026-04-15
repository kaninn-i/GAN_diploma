import os
import torch

from dataset_utils import (
    extract_crops,
    analyze_dataset,
    analyze_imbalance
)

from gan_training import train_gan
from augmentation import generate_objects, merge_datasets


def run_pipeline(
    dataset_path,
    crops_path,
    gan_weights_path,
    generated_path,
    output_dataset,
    epochs,
    generation_plan,
    model_type="dcgan"
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1 анализ датасета
    class_counts = analyze_dataset(dataset_path)
    imbalance = analyze_imbalance(class_counts)

    # 2 извлечение кропов
    crop_counts = extract_crops(dataset_path, crops_path)

    metrics = []

    # 3 генерация по классам
    for cls, num_generate in generation_plan.items():

        if num_generate == 0:
            continue

        class_data_path = os.path.join(
            crops_path,
            f"class_{cls}"
        )

        # если кропов нет — пропускаем
        if not os.path.exists(class_data_path):
            print(f"Нет кропов для class {cls}, пропускаем")
            continue

        if len(os.listdir(class_data_path)) == 0:
            print(f"Папка class {cls} пустая, пропускаем")
            continue

        class_weights = os.path.join(
            gan_weights_path,
            f"class_{cls}"
        )

        os.makedirs(class_weights, exist_ok=True)

        # train
        metrics_train = train_gan(
            data_path=class_data_path,
            save_path=class_weights,
            epochs=epochs,
            device=device,
            model_type=model_type
        )

        # generate
        class_output = os.path.join(
            generated_path,
            f"class_{cls}"
        )

        generate_objects(
            weights_path=os.path.join(class_weights, "generator.pth"),
            output_path=class_output,
            num_images=num_generate,
            device=device,
            model_type=model_type
        )

        metrics.append({
            "class": cls,
            "real": class_counts.get(cls, 0),
            "generated": num_generate,
            "g_loss": metrics_train["g_loss"],
            "d_loss": metrics_train["d_loss"]
        })

    # merge
    merge_datasets(
        original_dataset=dataset_path,
        generated_objects_path=generated_path,
        output_dataset=output_dataset
    )

    return {
        "class_counts": class_counts,
        "imbalance": imbalance,
        "metrics": metrics
    }