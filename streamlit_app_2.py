import streamlit as st
import os
import zipfile
import tempfile
from pathlib import Path
import pandas as pd

from dataset_utils import analyze_dataset, analyze_imbalance
from pipeline import run_pipeline

# ----- Настройка страницы -----
st.set_page_config(
    page_title="GAN Dataset Augmentation",
    page_icon="🎨",
    layout="wide"
)

# ----- Пользовательские стили (опционально) -----
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        font-weight: bold;
        background-color: #4CAF50;
        color: white;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----- Заголовок и описание -----
st.title("🎨 GAN Dataset Augmentation")
st.markdown("""
Загрузите ZIP‑архив с набором данных, который вы хотите расширить с помощью аугментации.  
Приложение проанализирует его на дисбаланс и сгенерирует дополнительные изображения с помощью нейронной сети **GAN**.
""")

# ----- Боковая панель с управлением -----
with st.sidebar:
    st.header("⚙️ Параметры генерации")
    st.markdown("---")

    # Выбор модели с подсказкой
    model_type = st.selectbox(
        "🧠 Архитектура GAN",
        ["dcgan", "ssd"],
        help="DCGAN — базовая модель, SSD — учитывает spatial‑признаки (рекомендуется для сложных сцен)"
    )

    epochs = st.slider(
        "⏱️ Эпох обучения",
        min_value=5,
        max_value=200,
        value=30,
        help="Больше эпох — лучше качество, но дольше генерация"
    )

    st.markdown("---")

# ----- Основная рабочая область -----
uploaded_zip = st.file_uploader(
    "📁 Загрузите ZIP‑архив с датасетом",
    type=["zip"],
    help="Структура архива: каждая папка — отдельный класс с изображениями внутри"
)

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "dataset")

        # Распаковка архива
        with zipfile.ZipFile(uploaded_zip) as z:
            z.extractall(dataset_path)

        # ----- Анализ датасета -----
        st.header("📊 Статистика исходного датасета")
        class_counts = analyze_dataset(dataset_path)
        imbalance = analyze_imbalance(class_counts)

        # Таблица с подсчётом классов
        df_stats = pd.DataFrame([
            {"Класс": cls, "Количество": count}
            for cls, count in class_counts.items()
        ])
        st.dataframe(df_stats, width='stretch')
        

        # Умное предупреждение о дисбалансе
        if imbalance:
            diff = imbalance['difference']
            min_class = imbalance['min_class']
            if diff > 0:
                st.warning(
                    f"⚠️ Обнаружен дисбаланс: класс **{min_class}** содержит на **{diff}** образцов меньше среднего. "
                    "Рекомендуется сгенерировать недостающее количество."
                )
            else:
                st.success("✅ Классы сбалансированы (или датасет содержит единственный класс).")
        else:
            st.info("ℹ️ Не удалось определить дисбаланс. Проверьте структуру датасета.")

        st.markdown("---")

        # ----- План генерации -----
        st.header("📋 План генерации")
        st.caption("Укажите, сколько синтетических изображений создать для каждого класса:")

        generation_plan = {}
        cols_plan = st.columns(len(class_counts))  # выводим поля в одну строку

        for idx, (cls, count) in enumerate(class_counts.items()):
            # Безопасно подставляем дефолтное значение, если дисбаланс есть и это проблемный класс
            default_val = 0
            if imbalance and cls == imbalance['min_class']:
                default_val = imbalance['difference']

            with cols_plan[idx]:
                generation_plan[cls] = st.number_input(
                    f"Класс **{cls}**",
                    min_value=0,
                    max_value=1000,
                    value=default_val,
                    step=10,
                    help=f"Текущее количество: {count}"
                )

        # Проверка: если ни один класс не выбран для генерации — показываем подсказку
        total_to_generate = sum(generation_plan.values())
        if total_to_generate == 0:
            st.info("💡 Выберите хотя бы один класс для генерации или измените значения.")

        # ----- Кнопка запуска -----
        st.markdown("---")
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            run_clicked = st.button("🚀 Запустить аугментацию", disabled=(total_to_generate == 0))

        if run_clicked:
            with st.spinner("⏳ Обучение GAN и генерация изображений... Это может занять несколько минут."):
                # Вызов основного пайплайна (без изменений)
                result = run_pipeline(
                    dataset_path=dataset_path,
                    crops_path=os.path.join(tmpdir, "crops"),
                    gan_weights_path=os.path.join(tmpdir, "weights"),
                    generated_path=os.path.join(tmpdir, "generated"),
                    output_dataset=os.path.join(tmpdir, "augmented"),
                    epochs=epochs,
                    generation_plan=generation_plan,
                    model_type=model_type
                )

            st.success("🎉 Аугментация завершена!")

            # ----- Метрики -----
            st.header("📈 Метрики генерации")
            if "metrics" in result and result["metrics"]:
                metrics_df = pd.DataFrame(result["metrics"])
                
                # Переименовываем колонки для красоты
                metrics_df = metrics_df.rename(columns={
                    "class": "Класс",
                    "real": "Реальных",
                    "generated": "Сгенерировано",
                    "g_loss": "G Loss",
                    "d_loss": "D Loss"
                })
                
                # Округляем значения потерь до 4 знаков
                metrics_df["G Loss"] = metrics_df["G Loss"].round(4)
                metrics_df["D Loss"] = metrics_df["D Loss"].round(4)
                
                # Отображаем таблицу
                st.dataframe(metrics_df, width='stretch', hide_index=True)
                
                # Краткая сводка
                total_generated = metrics_df["Сгенерировано"].sum()
                st.caption(f"Всего синтезировано изображений: **{total_generated}**")
            else:
                st.info("Метрики не были возвращены.")
                
            # --- Графики ---
            st.subheader("📈 Графики обучения GAN")

            for row in result["metrics"]:

                if "g_loss" not in row:
                    continue

                st.write(f"### Class {row['class']}")

                chart_data = {
                    "Generator": row["g_loss"],
                    "Discriminator": row["d_loss"]
                }

                st.line_chart(chart_data)

            # ----- Предпросмотр сгенерированных изображений -----
            st.header("🖼️ Примеры сгенерированных изображений")
            generated_dir = os.path.join(tmpdir, "generated")
            images = list(Path(generated_dir).rglob("*.png"))[:12]

            if images:
                cols_img = st.columns(6)
                for i, img_path in enumerate(images):
                    cols_img[i % 6].image(str(img_path), width='stretch')
            else:
                st.info("Изображения не найдены в папке generated/")

            # ----- Скачивание дополненного датасета -----
            st.header("📦 Скачать результат")
            augmented_dir = os.path.join(tmpdir, "augmented")
            if os.path.exists(augmented_dir):
                # Упаковываем в ZIP для скачивания
                zip_output = os.path.join(tmpdir, "augmented_dataset.zip")
                with zipfile.ZipFile(zip_output, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(augmented_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, augmented_dir)
                            zipf.write(file_path, arcname)

                with open(zip_output, "rb") as f:
                    st.download_button(
                        label="⬇️ Скачать дополненный датасет (ZIP)",
                        data=f,
                        file_name="augmented_dataset.zip",
                        mime="application/zip"
                    )
            else:
                st.warning("Папка с дополненным датасетом не найдена.")