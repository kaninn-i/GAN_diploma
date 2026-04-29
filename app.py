import streamlit as st
import torch
import tempfile
import zipfile
import os
from pathlib import Path
import pandas as pd
from augment_pipeline import run_full_pipeline
from data_utils import analyze_dataset

st.set_page_config(page_title="GAN Augmentation", layout="wide")
st.title("🎨 Аугментация датасета для обнаружения объектов")

uploaded_zip = st.file_uploader(
    "Загрузите ZIP с датасетом (любая структура папок, аннотации YOLO)",
    type="zip"
)

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(uploaded_zip) as z:
            z.extractall(tmpdir)

        # Если внутри только одна папка – заходим в неё
        entries = os.listdir(tmpdir)
        if len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0])):
            dataset_root = os.path.join(tmpdir, entries[0])
        else:
            dataset_root = tmpdir

        try:
            class_counts, image_objects = analyze_dataset(dataset_root)
            st.success(f"Найдено изображений с аннотациями: {len(image_objects)}")
            if class_counts:
                df = pd.DataFrame(
                    [{"Класс": cls, "Образцов": cnt} for cls, cnt in class_counts.items()]
                )
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("Классы не найдены. Проверьте разметку.")
        except ValueError as e:
            st.error(f"Ошибка чтения датасета: {e}")
            st.stop()

        # ==================== Боковая панель ====================
        with st.sidebar:
            st.header("⚙️ Параметры генерации")

            model_type = st.selectbox(
                "Архитектура GAN",
                options=["ssd", "dcgan"],
                index=0,
                help="SSD – StyleGAN2 Lite (лучше качество, медленнее). DCGAN – простая, быстрее."
            )

            img_size = st.selectbox(
                "Разрешение объектов",
                options=[64, 96, 128],
                index=1,    # 96 рекомендуемое
                help="Больше – качественнее, но дольше обучение и больше памяти."
            )

            epochs = st.slider(
                "Эпохи обучения",
                min_value=50,
                max_value=300,
                value=150,
                step=10,
                help="Для малых датасетов лучше 150–250 эпох."
            )

            st.markdown("---")
            st.subheader("Количество синтетических объектов")
            use_balance = st.checkbox("Автоматическая балансировка (дополнить до максимума)", value=True)
            if not use_balance:
                num_per_class = st.number_input(
                    "Сгенерировать объектов на каждый класс",
                    min_value=0, max_value=5000, value=50
                )
            else:
                num_per_class = None

            max_objs_per_img = st.slider(
                "Макс. объектов на новом изображении",
                min_value=1, max_value=5, value=3
            )

            blend_strength = st.slider(
                "Сила смешивания (0 – жёсткое, 1 – мягкое)",
                min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                help="0.0 – объект вставляется чётко, 1.0 – максимально плавно."
            )

            st.markdown("---")
            st.subheader("Разбиение на выборки")
            do_split = st.checkbox("Разбить итоговый датасет на train/val/test")
            split_config = None
            if do_split:
                train_pct = st.number_input("Train %", 0, 100, 70)
                val_pct = st.number_input("Validation %", 0, 100, 20)
                test_pct = st.number_input("Test %", 0, 100, 10)
                if train_pct + val_pct + test_pct != 100:
                    st.warning("Сумма процентов должна быть равна 100")
                else:
                    split_config = {
                        "train": train_pct / 100,
                        "val": val_pct / 100,
                        "test": test_pct / 100
                    }
                    split_config = {k: v for k, v in split_config.items() if v > 0}

            run = st.button("🚀 Запустить аугментацию")

        if run:
            if do_split and split_config is None:
                st.error("Исправьте проценты разбиения перед запуском.")
                st.stop()

            # Формируем план генерации
            generation_plan = None
            if not use_balance and num_per_class is not None:
                generation_plan = {cls: num_per_class for cls in class_counts}

            with st.spinner("Идёт обучение GAN и генерация объектов... Это может занять несколько минут."):
                output_tmp = os.path.join(tmpdir, "augmented_output")
                run_full_pipeline(
                    dataset_dir=dataset_root,
                    output_dir=output_tmp,
                    epochs=epochs,
                    balance_to_max=use_balance,
                    generation_plan_override=generation_plan,
                    max_objects_per_image=max_objs_per_img,
                    model_device="cuda" if torch.cuda.is_available() else "cpu",
                    padding_ratio=0.1,
                    split_config=split_config,
                    model_type=model_type,
                    img_size=img_size,
                    blend_strength=blend_strength
                )
            st.success("🎉 Аугментация завершена!")

            # Превью
            if do_split and split_config:
                preview_dir = os.path.join(output_tmp, "train", "images")
            else:
                preview_dir = os.path.join(output_tmp, "images")

            aug_imgs = list(Path(preview_dir).glob("*.jpg"))[:8]
            if aug_imgs:
                cols = st.columns(4)
                for i, p in enumerate(aug_imgs):
                    cols[i % 4].image(str(p), caption=p.name, use_container_width=True)

            # Скачивание
            zip_output = os.path.join(tmpdir, "augmented_dataset.zip")
            with zipfile.ZipFile(zip_output, "w") as zf:
                for root, _, files in os.walk(output_tmp):
                    for file in files:
                        full = os.path.join(root, file)
                        arcname = os.path.relpath(full, output_tmp)
                        zf.write(full, arcname)
            with open(zip_output, "rb") as f:
                st.download_button(
                    "📦 Скачать расширенный датасет",
                    f,
                    file_name="augmented_dataset.zip"
                )