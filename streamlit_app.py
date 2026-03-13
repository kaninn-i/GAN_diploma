import streamlit as st
import os

from pipeline import run_pipeline


st.title("GAN Dataset Augmentation")

st.write("Аугментация датасета с помощью GAN")

# параметры
dataset_path = st.text_input("Путь к датасету", "dataset")

epochs = st.slider(
    "Количество эпох GAN",
    min_value=5,
    max_value=200,
    value=30
)

images_per_class = st.slider(
    "Количество генерируемых изображений",
    min_value=10,
    max_value=500,
    value=50
)

start_button = st.button("Запустить pipeline")


if start_button:

    if not os.path.exists(dataset_path):
        st.error("Папка датасета не найдена")
    else:

        st.write("Запуск pipeline...")

        run_pipeline(
            dataset_path=dataset_path,
            epochs=epochs,
            images_per_class=images_per_class
        )

        st.success("Pipeline завершён")