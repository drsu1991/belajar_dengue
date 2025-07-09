import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('belajar_dengue_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('belajar_dengue_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Judul dan deskripsi
st.title("Prediksi Probabilitas Infeksi Dengue Berdasarkan Parameter Hematologi")

st.markdown("""
Aplikasi ini dikembangkan untuk memprediksi kemungkinan terjadinya Infeksi Dengue
Prediksi didasarkan pada data pemeriksaan hematologi rutin

Silakan masukkan hasil pemeriksaan laboratorium pasien untuk memperoleh estimasi probabilitas Dengue.
""")

with st.form("input_form"):
    st.subheader("Input Data Pemeriksaan Hematologi")

    Hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.0)
    Hematokrit = st.number_input("Hematokrit (%)", min_value=20.0, max_value=60.0, value=42.0)
    Leukosit = st.number_input("Leukosit (10³/uL)", min_value=0.5, max_value=30.0, value=6.0)
    Trombosit = st.number_input("Trombosit (10³/uL)", min_value=5.0, max_value=500.0, value=150.0)
    Neutrofil = st.number_input("Neutrofil (%)", min_value=0.0, max_value=100.0, value=55.0)
    Limfosit = st.number_input("Limfosit (%)", min_value=0.0, max_value=100.0, value=35.0)
    MPV = st.number_input("MPV (fL)", min_value=5.0, max_value=15.0, value=9.0)
    PDW = st.number_input("PDW", min_value=5.0, max_value=30.0, value=14.0)
    MCV = st.number_input("MCV (fL)", min_value=60.0, max_value=120.0, value=85.0)
    MCH = st.number_input("MCH (pg)", min_value=15.0, max_value=45.0, value=28.0)
    MCHC = st.number_input("MCHC (%)", min_value=20.0, max_value=40.0, value=33.0)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Data input
    input_data = pd.DataFrame({
        'Hemoglobin_g_dL': [Hemoglobin],
        'Hematokrit_percent': [Hematokrit],
        'Leukosit_10^3_uL': [Leukosit],
        'Trombosit_10^3_uL': [Trombosit],
        'Neutrofil_percent': [Neutrofil],
        'Limfosit_percent': [Limfosit],
        'MPV_fL': [MPV],
        'PDW': [PDW],
        'MCV_fL': [MCV],
        'MCH_pg': [MCH],
        'MCHC_percent': [MCHC]
    })

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi probabilitas
    prob = model.predict_proba(input_scaled)[0,1]
    pred_label = "POSITIF" if prob >=0.5 else "NEGATIF"

    st.subheader("Hasil Prediksi")
    st.write(f"**Estimasi Probabilitas Dengue:** `{prob:.2f}` (Threshold: 0.5)")
    st.write(f"**Interpretasi Prediksi:** `{pred_label}`")

    if prob >=0.5:
        st.warning("Hasil prediksi menunjukkan risiko Dengue yang tinggi. Pertimbangkan konfirmasi diagnosis dengan pemeriksaan klinis dan serologis.")
    else:
        st.success("Hasil prediksi menunjukkan risiko Dengue yang rendah. Tetap lakukan monitoring sesuai protokol klinis.")

# Footer identitas pengembang
st.markdown("""
---
**Pengembang:**  
dr. Suhendra Mandala Ernas  
**Institusi:**  
RSUD dr. Soetomo Surabaya
""")
