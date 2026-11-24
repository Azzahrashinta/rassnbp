from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Prediksi SNBP",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "xgboost.pkl"
DATASET_FILE = BASE_DIR / "dataset_pasangan.parquet"
MASTER_FILE = BASE_DIR / "DATA MASTER_SNBP 2023-2026.xlsx"
MASTER_SHEET = "Data SNBP"
SCHOOL_MAPPING_FILE = BASE_DIR / "mapping_sekolah.xlsx"

# Daftar 38 fitur mentah mengikuti urutan input model XGBoost.
FEATURE_COLUMNS = [
    "nilai_rapor_semester_1",
    "nilai_rapor_semester_2",
    "nilai_rapor_semester_3",
    "nilai_rapor_semester_4",
    "nilai_rapor_semester_5",
    "rata_rata_rapor",
    "tren_rapor",
    "peminatan",
    "peringkat_eligible",
    "jumlah_siswa_eligible",
    "posisi_relatif_eligible",
    "skor_z_rapor_dalam_sekolah",
    "jumlah_pemilih_prodi_sama",
    "jumlah_pemilih_ptn_sama",
    "peringkat_di_antara_pemilih_prodi_sama",
    "daya_tampung_tahun_ini",
    "daya_tampung_tahun_lalu",
    "keketatan_tahun_lalu",
    "tren_keketatan",
    "kelompok_prodi",
    "jenjang_prodi",
    "akreditasi_prodi",
    "wajib_portofolio",
    "prodi_baru",
    "nama_ptn",
    "provinsi_ptn",
    "kelinieran_jurusan",
    "kesesuaian_peminatan_kelompok_prodi",
    "ptn_satu_provinsi_dengan_sekolah",
    "akreditasi_sekolah",
    "jenis_sekolah",
    "provinsi_sekolah",
    "persentil_prestasi_sekolah",
    "rata_rata_rapor_sekolah_tahun_lalu",
    "simpangan_baku_rapor_sekolah_tahun_lalu",
    "proporsi_diterima_sekolah_tahun_lalu",
    "data_sekolah_tahun_lalu_tersedia",
    "urutan_pilihan",
]

SCHOOL_COLUMNS = [
    "akreditasi_sekolah",
    "jenis_sekolah",
    "provinsi_sekolah",
    "persentil_prestasi_sekolah",
    "rata_rata_rapor_sekolah_tahun_lalu",
    "simpangan_baku_rapor_sekolah_tahun_lalu",
    "proporsi_diterima_sekolah_tahun_lalu",
    "data_sekolah_tahun_lalu_tersedia",
    "jumlah_siswa_eligible",
]


# ============================================================
# 2. MEMUAT MODEL DAN DATA
# ============================================================

@st.cache_resource
def load_model(path_string):
    return joblib.load(path_string)


@st.cache_data

def load_dataset(path_string):
    data = pd.read_parquet(path_string)
    required = {
        "nama_siswa", "kode_sekolah", "tahun_angkatan", "peran_data",
        "diterima", "jumlah_siswa_eligible", "peminatan",
