import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Konfigurasi Tampilan Halaman (Ramah Mobile)
st.set_page_config(page_title="DSS Ketahanan Pangan Aceh", layout="centered")

# Header Aplikasi
st.title("🌾 DSS Analisis Prediktif Ketahanan Pangan")
st.subheader("Model Hybrid Diferensial Logistik – ARIMA")
st.write("Aplikasi analisis peramalan produksi komoditas padi berbasis data runtun waktu.")

# --- BARIS SAMPING (SIDEBAR) ---
st.sidebar.header("🎛️ Pengaturan & Input Data")

# Fitur Unggah File Excel atau CSV
st.sidebar.markdown("### 📅 Unggah Data Historis")
file_terunggah = st.sidebar.file_uploader("Pilih file Excel (.xlsx) atau CSV", type=["xlsx", "csv"])

# Pengaturan Parameter Model
st.sidebar.markdown("### Estimasi Awal Matematika")
guess_K = st.sidebar.number_input("Tebakan Carrying Capacity (K)", min_value=10, max_value=10000000, value=2500)
guess_A = st.sidebar.number_input("Tebakan Konstanta Integrasi (A)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
guess_r = st.sidebar.number_input("Tebakan Laju Pertumbuhan (r)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

st.sidebar.markdown("### Orde Model ARIMA")
arima_p = st.sidebar.slider("Orde Autoregressive (p)", 0, 3, 1)
arima_d = st.sidebar.slider("Orde Differencing (d)", 0, 2, 0)
arima_q = st.sidebar.slider("Orde Moving Average (q)", 0, 3, 1)

# --- MEMPROSES DATA ---
if file_terunggah is not None:
    try:
        if file_terunggah.name.endswith('.csv'):
            df_input = pd.read_csv(file_terunggah)
        else:
            df_input = pd.read_excel(file_terunggah)
        
        if 'Tahun' in df_input.columns and 'Produksi' in df_input.columns:
            df = df_input[['Tahun', 'Produksi']].dropna().copy()
            df = df.sort_values(by='Tahun').reset_index(drop=True)
            df['t'] = df['Tahun'] - df['Tahun'].min()
            df['Y_aktual'] = df['Produksi']
            st.success(f"✅ Sukses memuat file: {file_terunggah.name}")
        else:
            st.error("❌ Format file salah! Pastikan file memiliki kolom bernama 'Tahun' dan 'Produksi'.")
            st.stop()
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()
else:
    st.info("💡 Menampilkan data simulasi awal. Silakan unggah file Excel/CSV di panel samping untuk menganalisis data riil Anda.")
    tahun = np.arange(2015, 2026)
    t = year_t = tahun - 2015
    data_produksi = np.array([1500, 1650, 1800, 1920, 1980, 2050, 2020, 2100, 2180, 2220, 2270])
    df = pd.DataFrame({'Tahun': tahun, 't': t, 'Y_aktual': data_produksi})

# --- PROSES MATEMATIKA MODEL HYBRID ---
def model_logistik(t, K, A, r):
    return K / (1 + A * np.exp(-r * t))

try:
    tebakan_awal = [guess_K, guess_A, guess_r]
    params, _ = curve_fit(model_logistik, df['t'], df['Y_aktual'], p0=tebakan_awal, maxfev=10000)
    K_est, A_est, r_est = params
    df['P_logistik'] = model_logistik(df['t'], K_est, A_est, r_est)
    
    df['et_residual'] = df['Y_aktual'] - df['P_logistik']
    model_arima = ARIMA(df['et_residual'], order=(arima_p, arima_d, arima_q))
    hasil_arima = model_arima.fit()
    df['et_arima'] = hasil_arima.fittedvalues
    
    df['Y_hybrid'] = df['P_logistik'] + df['et_arima']
    
    rmse = np.sqrt(mean_squared_error(df['Y_aktual'], df['Y_hybrid']))
    mape = mean_absolute_percentage_error(df['Y_aktual'], df['Y_hybrid']) * 100

    # --- TAMPILAN UTAMA APLIKASI ---
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Carrying Capacity (K)", value=f"{K_est:.1f} Ton")
    col2.metric(label="Laju Pertumbuhan (r)", value=f"{r_est:.3f}")
    col3.metric(label="Akurasi Model (MAPE)", value=f"{mape:.2f}%")

    st.markdown("### 📈 Grafik Perbandingan Tren Produksi Padi")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Tahun'], df['Y_aktual'], 'ro-', label='Data Aktual')
    ax.plot(df['Tahun'], df['P_logistik'], 'g--', label='Tren Deterministik (Logistik)')
    ax.plot(df['Tahun'], df['Y_hybrid'], 'b*--', label='Hybrid Model (Logistik-ARIMA)')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Produksi (Ton)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("### 📋 Tabel Hasil Komputasi")
    st.dataframe(df[['Tahun', 'Y_aktual', 'P_logistik', 'et_residual', 'Y_hybrid']].style.format(precision=2))

except Exception as e:
    st.error(f"Gagal melakukan optimasi model. Silakan sesuaikan kembali nilai tebakan awal Anda pada sidebar. Error: {e}")
