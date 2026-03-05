import os
import random
import cv2


def insert_objects(
    original_dataset,
    generated_objects_path,
    output_dataset="dataset_augmented"
):
    images_path = os.path.join(original_dataset, "images")
    labels_path = os.path.join(original_dataset, "labels")

    out_images = os.path.join(output_dataset, "images")
    out_labels = os.path.join(output_dataset, "labels")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    image_files = os.listdir(images_path)

    for class_folder in os.listdir(generated_objects_path):
        class_id = int(class_folder.split("_")[-1])
        class_path = os.path.join(generated_objects_path, class_folder)

        for obj_file in os.listdir(class_path):

            # 1️⃣ Берём случайный фон
            bg_file = random.choice(image_files)
            bg_path = os.path.join(images_path, bg_file)
            label_path = os.path.join(labels_path, bg_file.replace(".jpg", ".txt"))

            background = cv2.imread(bg_path)
            h_bg, w_bg, _ = background.shape

            # 2️⃣ Загружаем объект
            obj_path = os.path.join(class_path, obj_file)
            obj = cv2.imread(obj_path)
            h_obj, w_obj, _ = obj.shape

            if h_obj >= h_bg or w_obj >= w_bg:
                continue

            # 3️⃣ Случайная позиция
            x_offset = random.randint(0, w_bg - w_obj)
            y_offset = random.randint(0, h_bg - h_obj)

            # 4️⃣ Вставка (простая, без alpha blending)
            background[y_offset:y_offset+h_obj, x_offset:x_offset+w_obj] = obj

            # 5️⃣ Новый bbox (YOLO формат)
            x_center = (x_offset + w_obj / 2) / w_bg
            y_center = (y_offset + h_obj / 2) / h_bg
            w_norm = w_obj / w_bg
            h_norm = h_obj / h_bg

            new_label_line = f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n"

            # 6️⃣ Сохраняем изображение
            new_img_name = f"aug_{class_id}_{obj_file}"
            cv2.imwrite(os.path.join(out_images, new_img_name), background)

            # 7️⃣ Копируем старую разметку + добавляем новую
            new_label_path = os.path.join(out_labels, new_img_name.replace(".png", ".txt").replace(".jpg", ".txt"))

            with open(new_label_path, "w") as f_out:

                # если у оригинала была разметка — копируем
                if os.path.exists(label_path):
                    with open(label_path, "r") as f_in:
                        f_out.writelines(f_in.readlines())

                f_out.write(new_label_line)

    print("Аугментация завершена.")