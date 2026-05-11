import os
import yaml
from ultralytics import YOLO


def create_dataset_yaml(dataset_dir, yaml_path, class_names=None):
    if class_names is None:
        class_names = ["object"]

    data = {
        "path": os.path.abspath(dataset_dir),
        "train": "train/images",
        "val": "val/images",
        "names": class_names
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    return yaml_path


def validate_yolo(
    dataset_dir,
    output_dir,
    class_names,
    model_name="yolov8n.pt",
    epochs=5,
    imgsz=640
):
    os.makedirs(output_dir, exist_ok=True)

    yaml_path = os.path.join(output_dir, "dataset.yaml")

    create_dataset_yaml(
        dataset_dir=dataset_dir,
        yaml_path=yaml_path,
        class_names=class_names
    )

    model = YOLO(model_name)

    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        project=output_dir,
        name="training",
        verbose=False,
        workers=0
    )

    metrics = model.val(
        data=yaml_path,
        split="val",
        imgsz=imgsz,
        workers=0,
        verbose=False
    )

    return {
        "map50": float(metrics.box.map50),
        "map5095": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr)
    }
