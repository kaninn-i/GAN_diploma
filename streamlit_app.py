import streamlit as st
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
import torch

from pipeline import run_pipeline

st.set_page_config(page_title="GAN Dataset Augmentation", layout="wide")
st.title("🎨 GAN-based Dataset Augmentation for Object Detection")

st.markdown("""
Загрузите ваш датасет в формате YOLO (папка с подкаталогами `images/` и `labels/`).
Поддерживается загрузка ZIP-архива, содержащего всю структуру.
""")

uploaded_zip = st.file_uploader(
    "📁 Выберите ZIP-архив с датасетом",
    type=["zip"],
    help="Архив должен содержать папки images/ и labels/ с файлами .jpg/.png и .txt соответственно."
)

col1, col2 = st.columns(2)
with col1:
    epochs = st.slider("🎯 Эпохи обучения GAN", 5, 200, 30, step=5)
with col2:
    num_images = st.slider("🖼️ Количество генерируемых изображений", 10, 500, 50, step=10)

device_option = st.radio(
    "💻 Устройство для вычислений",
    ["auto", "cpu", "cuda"],
    index=0,
    horizontal=True
)

start = st.button("🚀 Запустить аугментацию", type="primary", disabled=uploaded_zip is None)

if start and uploaded_zip is not None:
    # Определяем устройство
    if device_option == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_option
    st.info(f"Используется устройство: **{device.upper()}**")

    # Создаём временную директорию для распаковки
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "dataset")
        os.makedirs(dataset_path, exist_ok=True)

        # Распаковываем ZIP
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        st.success("✅ Архив успешно распакован")

        # Проверяем структуру датасета (должны быть папки images и labels)
        images_dir = os.path.join(dataset_path, "images")
        labels_dir = os.path.join(dataset_path, "labels")
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            st.error("❌ Неверная структура датасета! Ожидаются папки 'images' и 'labels' в корне архива.")
            st.stop()

        # Подсчёт исходных данных
        num_orig_images = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        num_orig_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
        st.write(f"📊 Исходный датасет: {num_orig_images} изображений, {num_orig_labels} разметок")

        # Контейнер для прогресса и логов
        progress_bar = st.progress(0, text="Подготовка...")
        log_placeholder = st.empty()

        # Переопределяем run_pipeline с выводом прогресса в Streamlit
        # Для этого лучше модифицировать pipeline.py, но чтобы не трогать основной код,
        # сделаем обёртку с ручным обновлением прогресса.

        try:
            # Этапы пайплайна с примерным весом
            steps = [
                ("Извлечение кропов объектов", 10),
                ("Обучение GAN", 50),
                ("Генерация синтетических изображений", 20),
                ("Сборка расширенного датасета", 20),
            ]

            current_progress = 0
            progress_bar.progress(current_progress, text=steps[0][0])

            # Шаг 1: извлечение кропов
            from dataset_utils import extract_crops
            crops_path = os.path.join(tmpdir, "crops")
            extract_crops(dataset_path, crops_path)
            current_progress += steps[0][1]
            progress_bar.progress(current_progress, text=steps[1][0])

            # Шаг 2: обучение GAN
            from gan_training import train_gan
            gan_weights_path = os.path.join(tmpdir, "gan_weights")
            train_gan(
                data_path=crops_path,
                save_path=gan_weights_path,
                epochs=epochs,
                device=device
            )
            current_progress += steps[1][1]
            progress_bar.progress(current_progress, text=steps[2][0])

            # Шаг 3: генерация
            from augmentation import generate_objects
            generated_path = os.path.join(tmpdir, "generated_objects")
            generate_objects(
                weights_path=os.path.join(gan_weights_path, "generator.pth"),
                output_path=generated_path,
                num_images=num_images,
                device=device,
                model_type="ssd" 
            )
            current_progress += steps[2][1]
            progress_bar.progress(current_progress, text=steps[3][0])

            # Шаг 4: сборка финального датасета
            from augmentation import merge_datasets
            output_dataset = os.path.join(tmpdir, "dataset_augmented")
            merge_datasets(
                original_dataset=dataset_path,
                generated_objects_path=generated_path,
                output_dataset=output_dataset
            )
            current_progress += steps[3][1]
            progress_bar.progress(current_progress, text="Завершено!")

            # Предпросмотр сгенерированных изображений
            st.subheader("🖼️ Примеры сгенерированных объектов")
            generated_files = sorted(Path(generated_path).glob("*.png"))[:10]
            if generated_files:
                cols = st.columns(5)
                for i, img_path in enumerate(generated_files):
                    cols[i % 5].image(str(img_path), width=150)
            else:
                st.warning("Нет сгенерированных изображений для предпросмотра")

            # Упаковка результата в ZIP для скачивания
            result_zip = os.path.join(tmpdir, "augmented_dataset.zip")
            with zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dataset):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, output_dataset)
                        zipf.write(full_path, arcname)

            # Подсчёт финального количества
            aug_images_dir = os.path.join(output_dataset, "images")
            aug_labels_dir = os.path.join(output_dataset, "labels")
            num_aug_images = len([f for f in os.listdir(aug_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            st.success(f"✅ Аугментация завершена! Итоговый датасет содержит {num_aug_images} изображений (+{num_aug_images - num_orig_images} синтетических)")

            # Кнопка скачивания
            with open(result_zip, "rb") as f:
                st.download_button(
                    label="📥 Скачать расширенный датасет (ZIP)",
                    data=f,
                    file_name="dataset_augmented.zip",
                    mime="application/zip"
                )

        except Exception as e:
            st.error(f"❌ Ошибка в процессе выполнения: {str(e)}")
            st.exception(e)