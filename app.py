import os
import zipfile
import tempfile
from pathlib import Path

import torch
import pandas as pd
import streamlit as st

from augment_pipeline import run_full_pipeline
from data_utils import analyze_dataset


st.set_page_config(page_title="GAN Dataset Augmentation", layout="wide")
st.title("🎨 Аугментация датасета для детекции объектов")


# =====================================================
# UI helpers
# =====================================================

def update_stage(stage_name, stage_num, stage_total):
    stage_progress.progress(stage_num / stage_total)
    stage_status.info(f"Этап {stage_num}/{stage_total}: {stage_name}")


def update_epoch(progress, message):
    epoch_progress.progress(progress)
    epoch_status.info(message)


# =====================================================
# Upload
# =====================================================

uploaded_zip = st.file_uploader(
    "Загрузите ZIP архив с датасетом (YOLO format)",
    type="zip"
)

if not uploaded_zip:
    st.stop()
    raise SystemExit(0)


with tempfile.TemporaryDirectory() as tmpdir:

    with zipfile.ZipFile(uploaded_zip) as z:
        z.extractall(tmpdir)

    entries = os.listdir(tmpdir)

    if len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0])):
        dataset_root = os.path.join(tmpdir, entries[0])
    else:
        dataset_root = tmpdir

    try:
        class_counts, image_objects = analyze_dataset(dataset_root)

    except Exception as e:
        st.error(f"Ошибка анализа датасета: {e}")
        st.stop()

    if not class_counts:
        st.error("Не удалось найти объекты в датасете.")
        st.stop()

    # =====================================================
    # Dataset diagnostics
    # =====================================================

    st.subheader("📊 Диагностика датасета")

    stats_df = pd.DataFrame(
        [
            {
                "Класс": f"class_{cls}",
                "Объектов": count
            }
            for cls, count in class_counts.items()
        ]
    )

    st.dataframe(stats_df, width='stretch')

    # =====================================================
    # Sidebar
    # =====================================================

    with st.sidebar:

        st.header("⚙️ Параметры генерации")

        model_type = st.selectbox(
            "Архитектура GAN",
            options=["ssd", "dcgan"],
            index=0
        )

        epochs = st.slider(
            "Эпохи обучения",
            min_value=10,
            max_value=150,
            value=50,
            step=5
        )

        st.divider()

        st.subheader("Балансировка")

        use_balance = st.checkbox(
            "Автоматически балансировать классы",
            value=True
        )

        generation_plan = None

        if not use_balance:

            generation_plan = {}

            st.caption("Выберите классы для аугментации")

            for cls in sorted(class_counts.keys()):

                col1, col2 = st.columns([1, 2])

                with col1:
                    enabled = st.checkbox(
                        f"class_{cls}",
                        key=f"class_enable_{cls}"
                    )

                with col2:

                    count = st.number_input(
                        "count",
                        min_value=0,
                        max_value=5000,
                        value=50,
                        step=10,
                        key=f"class_count_{cls}",
                        label_visibility="collapsed"
                    )

                if enabled:
                    generation_plan[cls] = count

        st.divider()

        max_objs_per_img = st.slider(
            "Макс. объектов на изображение",
            min_value=1,
            max_value=5,
            value=3
        )

        blend_strength = st.slider(
            "Сила смешивания",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

        st.divider()

        do_split = st.checkbox(
            "Разбить на train/val/test",
            value=True
        )

        split_config = None

        if do_split:

            train_pct = st.number_input("Train %", value=70)
            val_pct = st.number_input("Val %", value=20)
            test_pct = st.number_input("Test %", value=10)

            total_pct = train_pct + val_pct + test_pct

            if total_pct == 100:

                split_config = {
                    "train": train_pct / 100,
                    "val": val_pct / 100,
                    "test": test_pct / 100
                }

            else:
                st.warning("Сумма должна быть 100%")

        st.divider()

        run_yolo = st.checkbox(
            "YOLO validation",
            value=False
        )

        run_button = st.button(
            "🚀 Запустить аугментацию",
            width='stretch'
        )

    # =====================================================
    # Run
    # =====================================================

    if not run_button:
        st.stop()

    if do_split and split_config is None:
        st.error("Некорректное разбиение train/val/test")
        st.stop()

    if not use_balance and not generation_plan:
        st.error("Выберите хотя бы один класс для аугментации.")
        st.stop()

    # =====================================================
    # Runtime UI
    # =====================================================

    st.divider()
    st.subheader("⚡ Выполнение")

    stage_progress = st.progress(0)
    stage_status = st.empty()

    epoch_progress = st.progress(0)
    epoch_status = st.empty()

    # =====================================================
    # Run pipeline
    # =====================================================

    try:

        output_dir = os.path.join(tmpdir, "augmented_output")
        selected_img_size = 64 if model_type == "dcgan" else 96

        results = run_full_pipeline(
            dataset_dir=dataset_root,
            output_dir=output_dir,

            epochs=epochs,

            balance_to_max=use_balance,
            generation_plan_override=generation_plan,

            max_objects_per_image=max_objs_per_img,
            model_device="cuda" if torch.cuda.is_available() else "cpu",

            padding_ratio=0.1,
            split_config=split_config,

            model_type=model_type,
            img_size=selected_img_size,

            blend_strength=blend_strength,

            progress_callback=update_epoch,
            stage_callback=update_stage,

            run_yolo_validation=run_yolo
        )

    except Exception as e:

        st.error(f"Ошибка пайплайна: {e}")
        st.stop()

    # =====================================================
    # Results
    # =====================================================

    st.success("🎉 Аугментация завершена!")

    if "timings" in results:

        st.subheader("⏱ Время выполнения")

        timings_df = pd.DataFrame(
            [
                {
                    "Этап": name,
                    "Секунд": round(value, 2)
                }
                for name, value in results["timings"].items()
            ]
        )

        st.dataframe(timings_df, width='stretch')

    # =====================================================
    # YOLO
    # =====================================================

    yolo_results = results.get("yolo")

    if yolo_results and "error" not in yolo_results:

        st.subheader("🎯 YOLO Validation")

        comparison_df = pd.DataFrame([
            {
                "Метрика": "mAP50",
                "Original": round(yolo_results["original"]["map50"], 4),
                "Augmented": round(yolo_results["augmented"]["map50"], 4),
                "Delta": round(yolo_results["delta_map50"], 4)
            },
            {
                "Метрика": "Recall",
                "Original": round(yolo_results["original"]["recall"], 4),
                "Augmented": round(yolo_results["augmented"]["recall"], 4),
                "Delta": round(yolo_results["delta_recall"], 4)
            }
        ])

        st.dataframe(comparison_df, width='stretch')

    elif yolo_results and "error" in yolo_results:

        st.warning(
            f"YOLO validation завершился с ошибкой: {yolo_results['error']}"
        )

    # =====================================================
    # Preview
    # =====================================================

    st.subheader("Результат")

    preview_dir = os.path.join(output_dir, "train", "images")

    if not os.path.exists(preview_dir):
        preview_dir = os.path.join(output_dir, "images")

    preview_images = list(Path(preview_dir).glob("*.jpg"))[:8]

    if preview_images:

        cols = st.columns(4)

        for i, img_path in enumerate(preview_images):
            cols[i % 4].image(
                str(img_path),
                caption=img_path.name,
                width='stretch'
            )

    # =====================================================
    # Export
    # =====================================================

    zip_path = os.path.join(tmpdir, "augmented_dataset.zip")

    with zipfile.ZipFile(zip_path, "w") as zf:

        for root, _, files in os.walk(output_dir):

            for file in files:

                full_path = os.path.join(root, file)

                arcname = os.path.relpath(
                    full_path,
                    output_dir
                )

                zf.write(full_path, arcname)

    with open(zip_path, "rb") as f:

        st.download_button(
            "📦 Скачать датасет",
            f,
            file_name="augmented_dataset.zip",
            width='stretch'
        )
        
if __name__ == "__main__":
    pass