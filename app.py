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
st.title("Аугментация датасета для детекции объектов")


def update_stage(stage_name, stage_num, stage_total):
    stage_progress.progress(stage_num / stage_total)
    stage_status.info(f"Этап {stage_num}/{stage_total}: {stage_name}")


def update_epoch(progress, message):
    epoch_progress.progress(progress)
    epoch_status.info(message)


def get_class_label(cls_id, class_name_to_id):
    if class_name_to_id:
        id_to_name = {v: k for k, v in class_name_to_id.items()}
        return id_to_name.get(cls_id, f"class_{cls_id}")
    return f"class_{cls_id}"


def _preview_images(directory: str, n: int = 8) -> list[Path]:
    exts = {"*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"}
    all_imgs: list[Path] = []
    for pat in exts:
        all_imgs.extend(Path(directory).glob(pat))
    aug  = [p for p in all_imgs if p.stem.startswith("aug_")]
    rest = [p for p in all_imgs if not p.stem.startswith("aug_")]
    return (aug + rest)[:n]


# Загрузка датасета 

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

    # Диагностика 

    st.subheader("Диагностика датасета")

    stats_df = pd.DataFrame([
        {
            "Класс":    get_class_label(cls, class_name_to_id),
            "ID":       cls,
            "Объектов": count,
        }
        for cls, count in class_counts.items()
    ])
    st.dataframe(stats_df, width='stretch')

    total_objects = sum(class_counts.values())
    if total_objects < 120:
        st.info(
            "Объектов немного. Для стабильного GAN желательно **100–300+ кропов на класс** "
            "(включите несколько вариантов кропа в боковой панели)."
        )

    # Боковая панель 

    st.sidebar.title("Параметры генерации")

    with st.sidebar:

        _gan_options = {
            "dcgan":    "DCGAN",
            "dcgan_sn": "DCGAN + SN-D, без BN перед Tanh",
            "ssd":      "SSD — StyleGAN-like",
            "ssd_lite": "SSD lite — слабее noise, InstanceNorm в D",
        }
        model_type = st.selectbox(
            "Архитектура GAN",
            options=list(_gan_options.keys()),
            index=0,
            format_func=lambda k: _gan_options[k],
        )

        epochs = st.slider("Эпохи обучения", min_value=10, max_value=150, value=50, step=5)

        st.divider()
        st.subheader("Баланс классов")

        use_balance = st.checkbox("Автоматически балансировать классы", value=False)
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

        with st.expander("Дополнительно: тонкие настройки GAN"):

            if model_type in ("dcgan", "dcgan_sn"):
                selected_img_size = 64
                st.caption("Размер зафиксирован на **64×64** для DCGAN.")
            else:
                selected_img_size = st.select_slider(
                    "Размер кропа / выхода GAN (px)",
                    options=[64, 128],
                    value=64,
                    help="64 — быстрее и стабильнее для малых датасетов.",
                )

            n_critic       = st.slider("Шагов D на один шаг G", 1, 5, 1)
            r1_gamma       = st.slider("R1 регуляризация D (0 = выкл.)", 0.0, 30.0, 10.0, 0.5)
            save_best      = st.checkbox("Сохранять лучший чекпоинт по G-loss", value=True)
            compute_fid    = st.checkbox("Считать FID после генерации", value=True)
            log_experiment = st.checkbox("Лог в runs/", value=True)

            max_objs_per_img = st.slider("Макс. объектов на изображение", 1, 5, 3)

            blend_strength = st.slider(
                "Сила смешивания", 0.0, 1.0, 0.8, 0.1,
                help="1.0 — полный seamlessClone; <1.0 — линейное смешивание.",
            )

            crop_jitter_variants = st.slider(
                "Вариантов кропа на объект (jitter)", 1, 5, 3,
                help="Несколько смещений кропа без новых фото.",
            )

            with st.expander("Интеграция (фон без объектов)"):
                use_clean_background = st.checkbox(
                    "Удалять существующие bbox (inpaint) перед вставкой", value=True,
                )
                inpaint_dilate = st.slider("Дилатация маски bbox (px)", 0, 5, 2)
                inpaint_radius = st.slider("Радиус inpaint", 1, 10, 3)

        st.divider()

        run_yolo = st.checkbox("Валидация с помощью YOLO", value=False)

        run_button = st.button("Запустить аугментацию", width='stretch')

    # Валидация формы 

    if not run_button:
        st.stop()

    if do_split and split_config is None:
        st.error("Некорректное разбиение train/val/test")
        st.stop()

    if not use_balance and not generation_plan:
        st.error("Выберите хотя бы один класс для аугментации.")
        st.stop()

    st.divider()
    st.subheader("Выполнение")

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

    # Результаты 

    result_class_name_to_id = results.get("class_name_to_id", {})

    st.success("Аугментация завершена!")

    # Тайминги
    if "timings" in results:
        st.subheader("Время выполнения")
        timings_df = pd.DataFrame([
            {"Этап": name, "Секунд": round(v, 2)}
            for name, v in results["timings"].items()
        ])
        st.dataframe(timings_df, width='stretch')

    # GAN метрики
    gan_metrics = results.get("gan_metrics")
    if gan_metrics:
        st.subheader("Функции потерь GAN (последняя эпоха)")
        gm_rows = []
        for cls_id, m in sorted(gan_metrics.items(), key=lambda x: x[0]):
            gm_rows.append({
                "Класс":       get_class_label(cls_id, result_class_name_to_id),
                "G loss":      round(m.get("g_loss", 0), 4),
                "D loss":      round(m.get("d_loss", 0), 4),
                "Лучш. G":     round(m.get("best_g_loss", 0), 4),
                "Эпоха лучш.": m.get("best_epoch", -1),
                "Размер (px)": m.get("img_size_used", selected_img_size),
            })
        st.dataframe(pd.DataFrame(gm_rows), width='stretch')

    # FID
    gan_fid = results.get("gan_fid")
    if gan_fid:
        st.subheader("GAN: FID (кропы vs синтетика, ниже лучше)")
        fid_rows = [
            {"Класс": get_class_label(k, result_class_name_to_id), "FID": round(v, 2)}
            for k, v in sorted(gan_fid.items(), key=lambda x: x[0])
        ]
        st.dataframe(pd.DataFrame(fid_rows), width='stretch')

    # Графики обучения GAN
    epoch_histories = results.get("epoch_histories", {})
    if epoch_histories:
        with st.expander("Графики обучения GAN", expanded=False):
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

    # Превью синтетических объектов
    synth_dir = results.get("synth_dir", "")

    st.subheader("Сгенерированные изображения")

    if synth_dir and os.path.isdir(synth_dir):
        has_synth = False
        for class_subdir in sorted(os.listdir(synth_dir)):
            cls_path = os.path.join(synth_dir, class_subdir)
            if not os.path.isdir(cls_path):
                continue
            try:
                cls_id    = int(class_subdir.split("_")[1])
                cls_label = get_class_label(cls_id, result_class_name_to_id)
            except (IndexError, ValueError):
                cls_label = class_subdir

            synth_imgs = _preview_images(cls_path, n=8)
            if not synth_imgs:
                continue

            has_synth = True
            st.markdown(f"**{cls_label}**")
            cols = st.columns(4)
            for i, p in enumerate(synth_imgs):
                cols[i % 4].image(str(p), width='stretch')

        if not has_synth:
            st.info(
                "Синтетические объекты не найдены. "
                "Возможно, ни один класс не прошёл порог обучения."
            )
    else:
        st.info("Папка синтетических объектов не найдена.")

    # YOLO Валидация 

    yolo_results = results.get("yolo")

    if run_yolo:
        st.subheader("YOLO Валидация")

        if yolo_results is None:
            st.info("YOLO-валидация не выполнялась (нет разбивки на train/val или результат пуст).")
        elif "error" in yolo_results:
            st.error(f"Ошибка YOLO: {yolo_results['error']}")
        else:
            orig = yolo_results["original"]
            aug  = yolo_results["augmented"]

            comparison_df = pd.DataFrame([
                {
                    "Метрика":   "mAP50",
                    "Original":  orig["map50"],
                    "Augmented": aug["map50"],
                    "Delta":     yolo_results["delta_map50"],
                },
                {
                    "Метрика":   "mAP50-95",
                    "Original":  orig["map5095"],
                    "Augmented": aug["map5095"],
                    "Delta":     round(aug["map5095"] - orig["map5095"], 4),
                },
                {
                    "Метрика":   "Precision",
                    "Original":  orig["precision"],
                    "Augmented": aug["precision"],
                    "Delta":     round(aug["precision"] - orig["precision"], 4),
                },
                {
                    "Метрика":   "Recall",
                    "Original":  orig["recall"],
                    "Augmented": aug["recall"],
                    "Delta":     yolo_results["delta_recall"],
                },
            ])
            st.dataframe(comparison_df, width='stretch')

            col_m50, col_rec, col_pre, col_m5095 = st.columns(4)
            col_m50.metric(
                "Δ mAP50",
                f"{yolo_results['delta_map50']:+.4f}",
                delta_color="normal",
            )
            col_rec.metric(
                "Δ Recall",
                f"{yolo_results['delta_recall']:+.4f}",
                delta_color="normal",
            )
            col_pre.metric(
                "Δ Precision",
                f"{round(aug['precision'] - orig['precision'], 4):+.4f}",
                delta_color="normal",
            )
            col_m5095.metric(
                "Δ mAP50-95",
                f"{round(aug['map5095'] - orig['map5095'], 4):+.4f}",
                delta_color="normal",
            )

            orig_hist = orig.get("history", [])
            aug_hist  = aug.get("history", [])
            if orig_hist and aug_hist:
                with st.expander("Кривые обучения YOLO по эпохам", expanded=True):
                    col_left, col_right = st.columns(2)

                    def _history_chart(history: list[dict], title: str, container):
                        df = pd.DataFrame(history).set_index("epoch")
                        available = [c for c in ["map50", "recall", "precision"] if c in df.columns]
                        container.markdown(f"**{title}**")
                        container.line_chart(
                            df[available],
                            color=["#2ecc71", "#e67e22", "#3498db"][:len(available)],
                        )

                    _history_chart(orig_hist, "Original dataset",   col_left)
                    _history_chart(aug_hist,  "Augmented dataset",  col_right)

    # Экспорт 

    zip_path = os.path.join(tmpdir, "augmented_dataset.zip")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname   = os.path.relpath(full_path, output_dir)
                zf.write(full_path, arcname)

    with open(zip_path, "rb") as f:
        st.download_button(
            "Скачать аугментированный датасет",
            f,
            file_name="augmented_dataset.zip",
            width='stretch',
        )


if __name__ == "__main__":
    pass
