import os
import cv2


def yolo_to_bbox(label_line, img_w, img_h):
    cls, x, y, w, h = map(float, label_line.split())
    x1 = int((x - w/2) * img_w)
    y1 = int((y - h/2) * img_h)
    x2 = int((x + w/2) * img_w)
    y2 = int((y + h/2) * img_h)
    return int(cls), x1, y1, x2, y2


def extract_crops(dataset_path, output_path):
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(images_path):
        img_path = os.path.join(images_path, file)
        label_path = os.path.join(labels_path, file.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            cls, x1, y1, x2, y2 = yolo_to_bbox(line, w, h)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crop = cv2.resize(crop, (64, 64))

            cls_dir = os.path.join(output_path, f"class_{cls}")
            os.makedirs(cls_dir, exist_ok=True)

            save_path = os.path.join(cls_dir, f"{file[:-4]}_{i}.jpg")
            cv2.imwrite(save_path, crop)