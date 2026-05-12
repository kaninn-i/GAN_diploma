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


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def update_stage(stage_name, stage_num, stage_total):
    stage_progress.progress(stage_num / stage_total)
    stage_status.info(f"Этап {stage_num}/{stage_total}: {stage_name}")


def update_epoch(progress, message):
    epoch_progress.progress(progress)
    epoch_status.info(message)


def get_class_label(cls_id, class_name_to_id):
    """Читаемое имя класса: реальное (XML) или 'class_N' (YOLO)."""
    if class_name_to_id:
        id_to_name = {v: k for k, v in class_name_to_id.items()}
        return id_to_name.get(cls_id, f"class_{cls_id}")
    return f"class_{cls_id}"


def _preview_images(directory: str, n: int = 8) -> list[Path]:
    """
    Собирает до n изображений из directory.
    Приоритет — аугментированные (aug_*), потом любые остальные.
    """
    exts = {"*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"}
    all_imgs: list[Path] = []
    for pat in exts:
        all_imgs.extend(Path(directory).glob(pat))

    aug  = [p for p in all_imgs if p.stem.startswith("aug_")]
    rest = [p for p in all_imgs if not p.stem.startswith("aug_")]
    return (aug + rest)[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

uploaded_zip = st.file_uploader(
    "Загрузите ZIP архив с датасетом (YOLO .txt или Pascal VOC .xml)",
    type="zip",
)

if not uploaded_zip:
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:

    with zipfile.ZipFile(uploaded_zip) as z:
        z.extractall(tmpdir)

    entries = os.listdir(tmpdir)
    if len(entries) == 1 and os.path.isdir(os.path.join(tmpdir, entries[0])):
        dataset_root = os.path.join(tmpdir, entries[0])
    else:
        dataset_root = tmpdir

    try:
        class_counts, image_objects, class_name_to_id = analyze_dataset(dataset_root)
    except Exception as e:
        st.error(f"Ошибка анализа датасета: {e}")
        st.stop()

    if not class_counts:
        st.error("Не удалось найти объекты в датасете.")
        st.stop()

    fmt = "Pascal VOC (XML)" if class_name_to_id else "YOLO (TXT)"
    st.caption(f"Обнаружен формат: **{fmt}**")

    # ─────────────────────────────────────────────────────────────────────────
    # Диагностика датасета
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("📊 Диагностика датасета")

    stats_df = pd.DataFrame(
        [
            {
                "Класс": get_class_label(cls, class_name_to_id),
                "ID": cls,
                "Объектов": count,
            }
            for cls, count in class_counts.items()
        ]
    )

    st.dataframe(stats_df, use_container_width=True)

    total_objects = sum(class_counts.values())
    if total_objects < 120:
        st.info(
            "Объектов в датасете немного. Для стабильного GAN желательно **100–300+ кропов на класс** "
            "(включите **несколько вариантов кропа** в боковой панели). "
            "Чтобы сравнить архитектуры, запускайте прогоны с **одинаковыми эпохами** и одним архивом."
        )

    with st.expander("Сравнение архитектур GAN"):
        st.markdown(
            "1. Зафиксируйте один и тот же ZIP и число эпох.\n"
            "2. Прогоны с разными значениями **Архитектура GAN**.\n"
            "3. Сравните блок **GAN: FID** и визуально сгенерированные объекты.\n\n"
            "| Архитектура | Особенности |\n"
            "|---|---|\n"
            "| **ssd** | StyleGAN-like, mapping 4 слоя, BN в D, noise scale 1.0 |\n"
            "| **ssd_lite** | StyleGAN-like, mapping 2 слоя, InstanceNorm в D, noise scale 0.25 |\n"
            "| **dcgan** | DCGAN baseline, фиксирован на 64×64 |\n"
            "| **dcgan_sn** | DCGAN + spectral norm, без BN перед Tanh |\n"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar
    # ─────────────────────────────────────────────────────────────────────────

    with st.sidebar:

        st.header("⚙️ Параметры генерации")

        _gan_options = {
            "ssd":      "SSD — StyleGAN-like (baseline)",
            "ssd_lite": "SSD lite — слабее noise, InstanceNorm в D",
            "dcgan":    "DCGAN (baseline, 64×64)",
            "dcgan_sn": "DCGAN + SN-D, без BN перед Tanh (64×64)",
        }
        model_type = st.selectbox(
            "Архитектура GAN",
            options=list(_gan_options.keys()),
            index=0,
            format_func=lambda k: _gan_options[k],
        )

        # img_size — для DCGAN-вариантов фиксирован на 64
        if model_type in ("dcgan", "dcgan_sn"):
            selected_img_size = 64
            st.caption("Размер изображения зафиксирован на **64×64** для DCGAN.")
        else:
            selected_img_size = st.select_slider(
                "Размер кропа / выхода GAN (px)",
                options=[64, 128],
                value=64,
                help="64 — быстрее и стабильнее для малых датасетов; 128 — лучше деталь.",
            )

        epochs = st.slider(
            "Эпохи обучения",
            min_value=10,
            max_value=150,
            value=50,
            step=5,
        )

        st.divider()

        st.subheader("Балансировка")

        use_balance = st.checkbox("Автоматически балансировать классы", value=True)

        generation_plan = None

        if not use_balance:
            generation_plan = {}
            st.caption("Выберите классы для аугментации")
            for cls in sorted(class_counts.keys()):
                col1, col2 = st.columns([1, 2])
                with col1:
                    enabled = st.checkbox(
                        get_class_label(cls, class_name_to_id),
                        key=f"class_enable_{cls}",
                    )
                with col2:
                    count = st.number_input(
                        "count",
                        min_value=0,
                        max_value=5000,
                        value=50,
                        step=10,
                        key=f"class_count_{cls}",
                        label_visibility="collapsed",
                    )
                if enabled:
                    generation_plan[cls] = count

        st.divider()

        max_objs_per_img = st.slider(
            "Макс. объектов на изображение",
            min_value=1,
            max_value=5,
            value=3,
        )

        blend_strength = st.slider(
            "Сила Пуассонова смешивания",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="1.0 — полный seamlessClone; <1.0 — линейное смешивание с прямой вставкой.",
        )

        st.divider()

        do_split = st.checkbox("Разбить на train/val/test", value=True)

        split_config = None

        if do_split:
            train_pct = st.number_input("Train %", value=70)
            val_pct   = st.number_input("Val %",   value=20)
            test_pct  = st.number_input("Test %",  value=10)
            total_pct = train_pct + val_pct + test_pct
            if total_pct == 100:
                split_config = {
                    "train": train_pct / 100,
                    "val":   val_pct   / 100,
                    "test":  test_pct  / 100,
                }
            else:
                st.warning("Сумма должна быть 100%")

        st.divider()

        crop_jitter_variants = st.slider(
            "Вариантов кропа на объект (jitter)",
            min_value=1,
            max_value=5,
            value=3,
            help="Несколько смещений кропа без новых фото — больше данных для GAN.",
        )

        with st.expander("Дополнительно: обучение GAN"):
            n_critic    = st.slider("Шагов D на один шаг G (n_critic)", 1, 5, 1)
            r1_gamma    = st.slider("R1 регуляризация D (0 = выкл.)", 0.0, 30.0, 10.0, 0.5)
            save_best   = st.checkbox("Сохранять лучший чекпоинт по G-loss", value=True)
            compute_fid = st.checkbox("Считать FID после генерации", value=True)
            log_experiment = st.checkbox("Лог в runs/ (experiment_utils)", value=False)

        with st.expander("Интеграция (фон без объектов)"):
            use_clean_background = st.checkbox(
                "Удалять существующие bbox (inpaint) перед вставкой синтетики",
                value=True,
            )
            inpaint_dilate = st.slider("Дилатация маски bbox (px)", 0, 5, 2)
            inpaint_radius = st.slider("Радиус inpaint", 1, 10, 3)

        run_yolo = st.checkbox("YOLO validation", value=False)

        run_button = st.button(
            "🚀 Запустить аугментацию",
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────────────────────────────────

    if not run_button:
        st.stop()

    if do_split and split_config is None:
        st.error("Некорректное разбиение train/val/test")
        st.stop()

    if not use_balance and not generation_plan:
        st.error("Выберите хотя бы один класс для аугментации.")
        st.stop()

    st.divider()
    st.subheader("⚡ Выполнение")

    stage_progress = st.progress(0)
    stage_status   = st.empty()
    epoch_progress = st.progress(0)
    epoch_status   = st.empty()

    try:
        output_dir = os.path.join(tmpdir, "augmented_output")

        gan_train_kwargs = {
            "n_critic":  int(n_critic),
            "r1_gamma":  float(r1_gamma),
            "save_best": bool(save_best),
        }

        results = run_full_pipeline(
            dataset_dir=dataset_root,
            output_dir=output_dir,
            epochs=epochs,
            balance_to_max=use_balance,
            generation_plan_override=generation_plan,
            max_objects_per_image=max_objs_per_img,
            model_device="cuda" if torch.cuda.is_available() else "cpu",
            padding_ratio=0.1,
            crop_jitter_variants=crop_jitter_variants,
            crop_jitter_frac=0.15,
            split_config=split_config,
            model_type=model_type,
            img_size=selected_img_size,
            blend_strength=blend_strength,
            progress_callback=update_epoch,
            stage_callback=update_stage,
            run_yolo_validation=run_yolo,
            compute_fid=compute_fid,
            gan_train_kwargs=gan_train_kwargs,
            log_experiment=log_experiment,
            use_clean_background=use_clean_background,
            inpaint_dilate=int(inpaint_dilate),
            inpaint_radius=int(inpaint_radius),
        )

    except Exception as e:
        st.error(f"Ошибка пайплайна: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # Results
    # ─────────────────────────────────────────────────────────────────────────

    result_class_name_to_id = results.get("class_name_to_id", {})

    st.success("🎉 Аугментация завершена!")

    # ── Тайминги ──────────────────────────────────────────────────────────────
    if "timings" in results:
        st.subheader("⏱ Время выполнения")
        timings_df = pd.DataFrame(
            [{"Этап": name, "Секунд": round(v, 2)} for name, v in results["timings"].items()]
        )
        st.dataframe(timings_df, use_container_width=True)

    # ── GAN метрики ───────────────────────────────────────────────────────────
    gan_metrics = results.get("gan_metrics")
    if gan_metrics:
        st.subheader("GAN: лоссы (последняя эпоха)")
        gm_rows = []
        for cls_id, m in sorted(gan_metrics.items(), key=lambda x: x[0]):
            gm_rows.append(
                {
                    "Класс":      get_class_label(cls_id, result_class_name_to_id),
                    "G loss":     round(m.get("g_loss", 0), 4),
                    "D loss":     round(m.get("d_loss", 0), 4),
                    "Лучш. G":    round(m.get("best_g_loss", 0), 4),
                    "Эпоха лучш.": m.get("best_epoch", -1),
                    "Размер (px)": m.get("img_size_used", selected_img_size),
                }
            )
        st.dataframe(pd.DataFrame(gm_rows), use_container_width=True)

    # ── FID ───────────────────────────────────────────────────────────────────
    gan_fid = results.get("gan_fid")
    if gan_fid:
        st.subheader("GAN: FID (кропы vs синтетика, ниже лучше)")
        fid_rows = [
            {"Класс": get_class_label(k, result_class_name_to_id), "FID": round(v, 2)}
            for k, v in sorted(gan_fid.items(), key=lambda x: x[0])
        ]
        st.dataframe(pd.DataFrame(fid_rows), use_container_width=True)

    # ── Графики обучения GAN ─────────────────────────────────────────────────
    epoch_histories = results.get("epoch_histories", {})
    if epoch_histories:
        with st.expander("📈 Графики обучения GAN", expanded=False):
            for cls_id, history in sorted(epoch_histories.items()):
                if not history:
                    continue
                cls_label = get_class_label(cls_id, result_class_name_to_id)
                st.markdown(f"**Класс: {cls_label}**")
                chart_df = (
                    pd.DataFrame(history)
                    .set_index("epoch")
                    .rename(columns={"g_loss": "G loss", "d_loss": "D loss"})
                )
                st.line_chart(chart_df, color=["#e74c3c", "#3498db"])
                st.caption(
                    f"G loss финал: {history[-1]['g_loss']:.4f} | "
                    f"D loss финал: {history[-1]['d_loss']:.4f}"
                )

    # ── YOLO validation ───────────────────────────────────────────────────────
    yolo_results = results.get("yolo")
    if yolo_results and "error" not in yolo_results:
        st.subheader("🎯 YOLO Validation")
        comparison_df = pd.DataFrame(
            [
                {
                    "Метрика":   "mAP50",
                    "Original":  round(yolo_results["original"]["map50"], 4),
                    "Augmented": round(yolo_results["augmented"]["map50"], 4),
                    "Delta":     round(yolo_results["delta_map50"], 4),
                },
                {
                    "Метрика":   "Recall",
                    "Original":  round(yolo_results["original"]["recall"], 4),
                    "Augmented": round(yolo_results["augmented"]["recall"], 4),
                    "Delta":     round(yolo_results["delta_recall"], 4),
                },
            ]
        )
        st.dataframe(comparison_df, use_container_width=True)
    elif yolo_results and "error" in yolo_results:
        st.warning(f"YOLO validation завершился с ошибкой: {yolo_results['error']}")

    # ─────────────────────────────────────────────────────────────────────────
    # Preview: аугментированные изображения
    # ─────────────────────────────────────────────────────────────────────────

    st.subheader("🖼 Превью аугментированных изображений")

    # Определяем папку с результатами
    preview_dir = os.path.join(output_dir, "train", "images")
    if not os.path.isdir(preview_dir):
        preview_dir = os.path.join(output_dir, "images")

    if os.path.isdir(preview_dir):
        preview_images = _preview_images(preview_dir, n=8)
        if preview_images:
            cols = st.columns(4)
            for i, img_path in enumerate(preview_images):
                cols[i % 4].image(
                    str(img_path),
                    caption=img_path.name,
                    use_container_width=True,
                )
        else:
            st.info("Изображения не найдены в папке результатов.")
    else:
        st.info("Папка результатов не найдена.")

    # ── Превью синтетических объектов (GAN-выход) ─────────────────────────────
    synth_dir = results.get("synth_dir", "")
    if synth_dir and os.path.isdir(synth_dir):
        with st.expander("🔬 Синтетические объекты (выход GAN)", expanded=False):
            st.caption(
                "Здесь показаны объекты, напрямую сгенерированные GAN до вставки в фон. "
                "Если качество низкое — увеличьте число эпох или количество обучающих кропов."
            )
            for class_subdir in sorted(os.listdir(synth_dir)):
                cls_path = os.path.join(synth_dir, class_subdir)
                if not os.path.isdir(cls_path):
                    continue
                # Пробуем извлечь cls_id из имени папки class_N
                try:
                    cls_id = int(class_subdir.split("_")[1])
                    cls_label = get_class_label(cls_id, result_class_name_to_id)
                except (IndexError, ValueError):
                    cls_label = class_subdir

                st.markdown(f"**{cls_label}**")
                synth_imgs = _preview_images(cls_path, n=8)
                if synth_imgs:
                    cols = st.columns(4)
                    for i, p in enumerate(synth_imgs):
                        cols[i % 4].image(str(p), use_container_width=True)
                else:
                    st.caption("Нет изображений.")

    # ─────────────────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────────────────

    zip_path = os.path.join(tmpdir, "augmented_dataset.zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname   = os.path.relpath(full_path, output_dir)
                zf.write(full_path, arcname)

    with open(zip_path, "rb") as f:
        st.download_button(
            "📦 Скачать аугментированный датасет",
            f,
            file_name="augmented_dataset.zip",
            use_container_width=True,
        )


if __name__ == "__main__":
    pass
