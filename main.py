import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Prediksi Harga Laptop", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_laptop.joblib")
META_PATH = os.path.join(BASE_DIR, "metadata_laptop.joblib")

def baca_dataset():
    kandidat = [
        os.path.join(BASE_DIR, "laptop_price.csv"),
        os.path.join(BASE_DIR, "laptop_price - dataset.csv"),
    ]
    csv_path = next((p for p in kandidat if os.path.exists(p)), None)
    if csv_path is None:
        st.error(f"File dataset CSV tidak ditemukan di folder: {BASE_DIR}")
        st.stop()

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    df.columns = [c.strip() for c in df.columns]

    # Otomatis ekstrak kolom jika CSV memakai format mentah Kaggle
    if "Cpu" in df.columns and "CPU_Company" not in df.columns:
        df["CPU_Company"] = df["Cpu"].astype(str).apply(lambda x: x.split()[0].strip())
        df["CPU_Frequency (GHz)"] = (
            df["Cpu"].astype(str)
            .str.extract(r"(\d+(?:\.\d+)?)\s*GHz", expand=False)
            .astype(float)
        )
        df["CPU_Type"] = (
            df["Cpu"].astype(str)
            .str.replace(r"\s*\d+(?:\.\d+)?\s*GHz$", "", regex=True)
            .apply(lambda x: " ".join(x.split()[1:]).strip())
        )
    if "Ram" in df.columns and "RAM (GB)" not in df.columns:
        df["RAM (GB)"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).str.strip().astype(int)
    if "Gpu" in df.columns and "GPU_Company" not in df.columns:
        df["GPU_Company"] = df["Gpu"].astype(str).apply(lambda x: x.split()[0].strip())
        df["GPU_Type"] = df["Gpu"].astype(str).apply(lambda x: " ".join(x.split()[1:]).strip())
    if "Weight" in df.columns and "Weight (kg)" not in df.columns:
        df["Weight (kg)"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).str.strip().astype(float)
    if "Price_euros" in df.columns and "Price (Euro)" not in df.columns:
        df["Price (Euro)"] = df["Price_euros"].astype(float)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df

@st.cache_resource
def muat_model_dan_meta():
    kolom_wajib = [
        "Company", "Product", "product_by_company", "TypeName", "Inches",
        "ScreenResolution", "CPU_Company", "CPU_Type", "CPU_Frequency (GHz)",
        "RAM (GB)", "Memory", "GPU_Company", "GPU_Type", "OpSys", "Weight (kg)"
    ]
    try:
        model = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH)
        if all(k in meta for k in kolom_wajib):
            return model, meta
    except Exception:
        pass

    df = baca_dataset()

    fitur_kategorikal = [
        "Company", "Product", "TypeName", "ScreenResolution",
        "CPU_Company", "CPU_Type", "Memory", "GPU_Company",
        "GPU_Type", "OpSys"
    ]
    fitur_numerik = ["Inches", "CPU_Frequency (GHz)", "RAM (GB)", "Weight (kg)"]

    df = df.dropna(subset=fitur_kategorikal + fitur_numerik + ["Price (Euro)"])
    X = df[fitur_kategorikal + fitur_numerik]
    y = df["Price (Euro)"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), fitur_numerik),
        ("cat", OneHotEncoder(handle_unknown="ignore"), fitur_kategorikal)
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42))
    ])
    model.fit(X_train, y_train)

    product_by_company = {
        comp: sorted(df[df["Company"] == comp]["Product"].unique().tolist())
        for comp in sorted(df["Company"].unique().tolist())
    }
    meta = {
        "Company": sorted(df["Company"].unique().tolist()),
        "Product": sorted(df["Product"].unique().tolist()),
        "product_by_company": product_by_company,
        "TypeName": sorted(df["TypeName"].unique().tolist()),
        "Inches": sorted(df["Inches"].unique().tolist()),
        "ScreenResolution": sorted(df["ScreenResolution"].unique().tolist()),
        "CPU_Company": sorted(df["CPU_Company"].unique().tolist()),
        "CPU_Type": sorted(df["CPU_Type"].unique().tolist()),
        "CPU_Frequency (GHz)": sorted(df["CPU_Frequency (GHz)"].unique().tolist()),
        "RAM (GB)": sorted(df["RAM (GB)"].unique().tolist()),
        "Memory": sorted(df["Memory"].unique().tolist()),
        "GPU_Company": sorted(df["GPU_Company"].unique().tolist()),
        "GPU_Type": sorted(df["GPU_Type"].unique().tolist()),
        "OpSys": sorted(df["OpSys"].unique().tolist()),
        "Weight (kg)": sorted(df["Weight (kg)"].unique().tolist())
    }

    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(meta, META_PATH)
    except Exception:
        pass

    return model, meta

model, meta = muat_model_dan_meta()

st.title("Prediksi Harga Laptop")
st.write("Pilih spesifikasi laptop dan atur kurs Rupiah di bawah ini untuk menghitung estimasi harga.")

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox("Merek Laptop", meta["Company"])
with col2:
    product = st.selectbox("Seri / Model Laptop", meta["product_by_company"][company])

col3, col4 = st.columns(2)
with col3:
    typename = st.selectbox("Tipe Laptop", meta["TypeName"])
with col4:
    inches = st.selectbox("Ukuran Layar (Inci)", meta["Inches"])

col5, col6 = st.columns(2)
with col5:
    screen_res = st.selectbox("Resolusi Layar", meta["ScreenResolution"])
with col6:
    opsys = st.selectbox("Sistem Operasi", meta["OpSys"])

col7, col8 = st.columns(2)
with col7:
    cpu_company = st.selectbox("Merek Prosesor", meta["CPU_Company"])
with col8:
    cpu_type = st.selectbox("Tipe Prosesor", meta["CPU_Type"])

col9, col10 = st.columns(2)
with col9:
    cpu_freq = st.selectbox("Kecepatan Prosesor (GHz)", meta["CPU_Frequency (GHz)"])
with col10:
    ram = st.selectbox("Kapasitas RAM (GB)", meta["RAM (GB)"])

col11, col12 = st.columns(2)
with col11:
    memory = st.selectbox("Kapasitas Penyimpanan", meta["Memory"])
with col12:
    weight = st.selectbox("Berat Laptop (kg)", meta["Weight (kg)"])

col13, col14 = st.columns(2)
with col13:
    gpu_company = st.selectbox("Merek Kartu Grafis", meta["GPU_Company"])
with col14:
    gpu_type = st.selectbox("Tipe Kartu Grafis", meta["GPU_Type"])

col15, col16 = st.columns(2)
with col15:
    kurs_idr = st.number_input(
        "Kurs Rupiah per 1 USD (IDR)",
        min_value=1000.0,
        value=16200.0,
        step=100.0,
        format="%.0f"
    )

if st.button("Prediksi Harga Laptop", use_container_width=True):
    data = pd.DataFrame([{
        "Company": company,
        "Product": product,
        "TypeName": typename,
        "Inches": inches,
        "ScreenResolution": screen_res,
        "CPU_Company": cpu_company,
        "CPU_Type": cpu_type,
        "CPU_Frequency (GHz)": cpu_freq,
        "RAM (GB)": ram,
        "Memory": memory,
        "GPU_Company": gpu_company,
        "GPU_Type": gpu_type,
        "OpSys": opsys,
        "Weight (kg)": weight
    }])

    harga_euro = model.predict(data)[0]
    harga_dolar = harga_euro * 1.08
    harga_rupiah = harga_dolar * kurs_idr

    hasil1, hasil2 = st.columns(2)
    with hasil1:
        st.success(f"Estimasi Harga (USD): $ {harga_dolar:,.2f}")
    with hasil2:
        st.info(f"Estimasi Harga (IDR): Rp {harga_rupiah:,.0f}")