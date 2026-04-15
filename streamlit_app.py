import streamlit as st
import os
import zipfile
import tempfile
from pathlib import Path
import pandas as pd

from dataset_utils import analyze_dataset, analyze_imbalance
from pipeline import run_pipeline

st.set_page_config(layout="wide")
st.title("GAN Dataset Augmentation")

uploaded_zip = st.file_uploader("Dataset ZIP", type=["zip"])

model_type = st.selectbox(
    "Model",
    ["dcgan", "ssd"]
)

epochs = st.slider("Epochs", 5, 200, 30)

if uploaded_zip:

    with tempfile.TemporaryDirectory() as tmpdir:

        dataset_path = os.path.join(tmpdir, "dataset")

        with zipfile.ZipFile(uploaded_zip) as z:
            z.extractall(dataset_path)

        # =====================
        # Анализ датасета
        # =====================

        class_counts = analyze_dataset(dataset_path)
        imbalance = analyze_imbalance(class_counts)

        st.subheader("Dataset statistics")

        df = pd.DataFrame([
            {"class": k, "count": v}
            for k, v in class_counts.items()
        ])

        st.dataframe(df)

        if imbalance:
            st.warning(
                f"Imbalance detected: class {imbalance['min_class']} "
                f"needs +{imbalance['difference']} samples"
            )

        # =====================
        # генерация настройки
        # =====================

        st.subheader("Generation per class")

        generation_plan = {}

        for cls, count in class_counts.items():
            generation_plan[cls] = st.number_input(
                f"class {cls}",
                min_value=0,
                max_value=1000,
                value=imbalance["difference"] if cls == imbalance["min_class"] else 0
            )

        if st.button("Run augmentation"):

            with st.spinner("Running pipeline..."):

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

            st.success("Done")

            # =====================
            # Метрики
            # =====================

            st.subheader("Generation metrics")

            metrics_df = pd.DataFrame(result["metrics"])
            st.dataframe(metrics_df)

            # =====================
            # preview
            # =====================

            st.subheader("Generated preview")

            generated_dir = os.path.join(tmpdir, "generated")

            images = list(Path(generated_dir).rglob("*.png"))[:12]

            cols = st.columns(6)

            for i, img in enumerate(images):
                cols[i % 6].image(str(img))