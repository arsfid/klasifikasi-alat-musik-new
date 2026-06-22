# app.py — Streamlit App Klasifikasi Alat Musik

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from PIL import Image

# ── Load model & class names ──
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("instrument_model.keras")
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, class_names

model, CLASS_NAMES = load_resources()


# ── Fungsi Prediksi ──
def predict(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]

    # Urutkan semua prediksi dari terbesar ke terkecil
    sorted_idx = np.argsort(preds)[::-1]

    hasil = [
        (CLASS_NAMES[i], round(float(preds[i]) * 100, 2))
        for i in sorted_idx
    ]

    return hasil


# ── UI Streamlit ──
st.set_page_config(
    page_title="Klasifikasi Alat Musik",
    page_icon="🎵"
)

st.title("🎵 Klasifikasi Alat Musik")
st.write("Upload gambar alat musik untuk diprediksi.")

uploaded = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded)

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    with st.spinner("Memproses..."):
        hasil = predict(image)

    # Prediksi terbaik
    kelas, conf = hasil[0]

    st.subheader("🎯 Prediksi Utama")
    st.success(
        f"{kelas.upper()} — Confidence: {conf:.2f}%"
    )

    # Top 3 prediksi
    st.subheader("🏆 Top 3 Prediksi")

    for rank, (nama, c) in enumerate(hasil[:3], 1):
        st.progress(
            int(c),
            text=f"#{rank} {nama} — {c:.2f}%"
        )

    # Semua prediksi
    st.subheader("📋 Semua Prediksi")

    df = pd.DataFrame(
        hasil,
        columns=["Alat Musik", "Confidence (%)"]
    )

    df.index += 1

    st.dataframe(
        df,
        use_container_width=True
    )