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

# Daftar fitur mengikuti 37 fitur yang digunakan pada model XGBoost.
FEATURE_COLUMNS = [
    "nilai_rapor_semester_1",
    "nilai_rapor_semester_2",
    "nilai_rapor_semester_3",
    "nilai_rapor_semester_4",
    "nilai_rapor_semester_5",
    "rata_rata_rapor",
    "tren_rapor",
    "peminatan",
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
        *SCHOOL_COLUMNS,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Kolom dataset tidak ditemukan: {sorted(missing)}")
    return data


@st.cache_data

def load_master(path_string, sheet_name):
    data = pd.read_excel(path_string, sheet_name=sheet_name)
    data.columns = data.columns.str.strip()
    required = {
        "Kode Prodi", "Universitas", "Prodi/Jurusan", "Kelompok", "Jenjang",
        "Daya Tampung", "Keketatan", "Portofolio", "Akreditasi", "Provinsi", "Tahun",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Kolom master SNBP tidak ditemukan: {sorted(missing)}")
    return data


def parse_percentage(value):
    """Mengubah teks seperti '12.47%' menjadi 0.1247."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return number / 100 if number > 1 else number
    text = str(value).strip().replace("%", "").replace(",", ".")
    try:
        return float(text) / 100
    except ValueError:
        return np.nan


def clean_text(value):
    """Meniru pembersihan teks fitur: trim dan huruf kapital."""
    if pd.isna(value):
        return np.nan
    return str(value).strip().upper()


def prepare_master(master_raw):
    """Menyiapkan kolom master dengan aturan kategori yang digunakan saat feature engineering."""
    master = master_raw.copy()
    master["Tahun"] = pd.to_numeric(master["Tahun"], errors="coerce").astype("Int64")
    master["Kode Prodi"] = pd.to_numeric(master["Kode Prodi"], errors="coerce")
    master["Daya Tampung"] = pd.to_numeric(master["Daya Tampung"], errors="coerce")
    master["keketatan_decimal"] = master["Keketatan"].map(parse_percentage)

    for col in ["Universitas", "Prodi/Jurusan", "Kelompok", "Jenjang", "Akreditasi", "Provinsi", "Portofolio"]:
        master[col] = master[col].map(clean_text)

    accreditation_map = {
        "UNGGUL": "UNGGUL",
        "A": "UNGGUL",
        "TERAKREDITASI UNGGUL": "UNGGUL",
        "ACCREDITED (ABET)": "UNGGUL",
        "BAIK SEKALI": "BAIK SEKALI",
        "B": "BAIK SEKALI",
        "BAIK": "BAIK",
        "C": "BAIK",
    }
    master["akreditasi_model"] = (
        master["Akreditasi"].map(accreditation_map).fillna("BELUM ATAU SEMENTARA")
    )
    master["wajib_portofolio_model"] = np.where(
        master["Portofolio"].isna(), np.nan,
        (master["Portofolio"] != "TIDAK ADA").astype(int),
    )
    return master


def make_program_reference(master, target_year):
    """Membentuk fitur prodi mengikuti pergeseran tahun pada notebook dataset.

    Untuk target 2027, master 2026 menjadi sumber daya tampung terbaru.
    Keketatan menggunakan tahun terakhir sebelum target yang memiliki data peminat.
    """
    available_years = sorted(master["Tahun"].dropna().astype(int).unique().tolist())
    eligible_capacity_years = [year for year in available_years if year <= target_year]
    if not eligible_capacity_years:
        raise ValueError("Tidak ada data daya tampung untuk tahun target atau sebelumnya.")

    capacity_year = target_year if target_year in available_years else max(eligible_capacity_years)
    current = master[master["Tahun"] == capacity_year].copy()
    if current.empty:
        raise ValueError(f"Data master tahun {capacity_year} tidak ditemukan.")

    # Pada target 2027, tahun t-1 adalah 2026. Data ini juga menjadi kapasitas terbaru.
    previous_capacity_year = target_year - 1
    previous = master[master["Tahun"] == previous_capacity_year][["Kode Prodi", "Daya Tampung"]].copy()
    previous = previous.rename(columns={"Daya Tampung": "daya_tampung_tahun_lalu"})

    years_with_competitiveness = sorted(
        master.loc[
            (master["Tahun"].astype("float") < target_year)
            & master["keketatan_decimal"].notna(),
            "Tahun",
        ].dropna().astype(int).unique().tolist()
    )
    if not years_with_competitiveness:
        keketatan_year = None
        keketatan_before_year = None
        last_comp = pd.DataFrame(columns=["Kode Prodi", "keketatan_tahun_lalu"])
        before_comp = pd.DataFrame(columns=["Kode Prodi", "keketatan_sebelumnya"])
    else:
        keketatan_year = max(years_with_competitiveness)
        before_years = [year for year in years_with_competitiveness if year < keketatan_year]
        keketatan_before_year = max(before_years) if before_years else None
        last_comp = master[master["Tahun"] == keketatan_year][["Kode Prodi", "keketatan_decimal"]].rename(
            columns={"keketatan_decimal": "keketatan_tahun_lalu"}
        )
        if keketatan_before_year is None:
            before_comp = pd.DataFrame(columns=["Kode Prodi", "keketatan_sebelumnya"])
        else:
            before_comp = master[master["Tahun"] == keketatan_before_year][["Kode Prodi", "keketatan_decimal"]].rename(
                columns={"keketatan_decimal": "keketatan_sebelumnya"}
            )

    program = current[[
        "Kode Prodi", "Universitas", "Prodi/Jurusan", "Kelompok", "Jenjang",
        "akreditasi_model", "wajib_portofolio_model", "Provinsi", "Daya Tampung",
    ]].rename(columns={
        "Kode Prodi": "kode_prodi",
        "Universitas": "nama_ptn",
        "Prodi/Jurusan": "nama_prodi",
        "Kelompok": "kelompok_prodi",
        "Jenjang": "jenjang_prodi",
        "akreditasi_model": "akreditasi_prodi",
        "wajib_portofolio_model": "wajib_portofolio",
        "Provinsi": "provinsi_ptn",
        "Daya Tampung": "daya_tampung_tahun_ini",
    })
    program = program.merge(
        previous.rename(columns={"Kode Prodi": "kode_prodi"}),
        on="kode_prodi", how="left", validate="one_to_one",
    )
    program = program.merge(
        last_comp.rename(columns={"Kode Prodi": "kode_prodi"}),
        on="kode_prodi", how="left", validate="one_to_one",
    )
    program = program.merge(
        before_comp.rename(columns={"Kode Prodi": "kode_prodi"}),
        on="kode_prodi", how="left", validate="one_to_one",
    )
    program["tren_keketatan"] = (
        program["keketatan_tahun_lalu"] - program["keketatan_sebelumnya"]
    )
    program["prodi_baru"] = program["daya_tampung_tahun_lalu"].isna().astype(int)
    program = program.drop(columns="keketatan_sebelumnya")
    program["kode_prodi"] = pd.to_numeric(program["kode_prodi"], errors="coerce")

    metadata = {
        "capacity_year": int(capacity_year),
        "previous_capacity_year": int(previous_capacity_year),
        "keketatan_year": keketatan_year,
        "keketatan_before_year": keketatan_before_year,
    }
    return program, metadata


def build_school_reference(data):
    """Satu baris referensi per sekolah-angkatan dari dataset pasangan."""
    school_ref = (
        data.sort_values(["kode_sekolah", "tahun_angkatan"])
        .groupby(["kode_sekolah", "tahun_angkatan"], as_index=False)
        .first()
    )
    return school_ref


def get_school_record(school_ref, school_code, target_year):
    records = school_ref[school_ref["kode_sekolah"] == school_code].copy()
    if records.empty:
        raise ValueError("Informasi sekolah tidak ditemukan di dataset.")

    # Gunakan angkatan target bila ada; jika belum ada, gunakan angkatan terbaru yang tersedia.
    past_years = records[records["tahun_angkatan"] <= target_year]["tahun_angkatan"]
    reference_year = int(past_years.max()) if not past_years.empty else int(records["tahun_angkatan"].max())
    row = records[records["tahun_angkatan"] == reference_year].iloc[0].copy()
    return row, reference_year


def compute_report_features(scores):
    """Rata-rata dan slope tren rapor mengikuti rumus notebook pembentukan dataset."""
    semester_numbers = np.arange(1, 6, dtype=float)
    values = np.array(scores, dtype=float)
    mask = np.isfinite(values)

    average = float(np.mean(values[mask])) if mask.any() else np.nan
    if mask.sum() >= 2:
        trend = float(np.polyfit(semester_numbers[mask], values[mask], 1)[0])
    else:
        trend = np.nan
    return average, trend


def compute_match_features(specialization, program_group, school_province, ptn_province):
    """Aturan kesesuaian peminatan dan kesamaan provinsi mengikuti notebook dataset."""
    if pd.isna(program_group):
        match = np.nan
    elif specialization == "ALL" or program_group == "VOKASI":
        match = "TIDAK BERLAKU"
    else:
        pairs = {("IPA", "SAINTEK"), ("IPS", "SOSHUM")}
        match = "SESUAI" if (specialization, program_group) in pairs else "TIDAK SESUAI"

    if pd.isna(ptn_province):
        same_province = np.nan
    else:
        provinces = [part.strip().upper() for part in str(ptn_province).split(";")]
        same_province = int(str(school_province).strip().upper() in provinces)
    return match, same_province


def build_choice_row(
    scores,
    specialization,
    eligible_rank,
    eligible_total,
    school_record,
    program_record,
    linearity,
    choice_order,
    target_year,
    school_reference_year,
):
    average, trend = compute_report_features(scores)
    match, same_province = compute_match_features(
        specialization,
        program_record["kelompok_prodi"],
        school_record["provinsi_sekolah"],
        program_record["provinsi_ptn"],
    )

    # Statistik persaingan satu angkatan tidak dapat direkonstruksi secara utuh
    # dari dataset_pasangan.parquet karena file ini tidak memuat seluruh roster siswa
    # dan seluruh pilihan asli. Pipeline training akan menangani NaN melalui imputer.
    z_score = np.nan
    same_program_competitors = np.nan
    same_university_competitors = np.nan
    rank_among_same_program = np.nan

    # Fitur historis sekolah pada dataset 2026 merujuk ke 2025.
    # Untuk prediksi 2027, data angkatan 2026 lengkap tidak tersedia pada file ini,
    # sehingga tidak memakai statistik 2025 seolah-olah statistik 2026.
    if target_year == school_reference_year:
        school_history_mean = school_record.get("rata_rata_rapor_sekolah_tahun_lalu", np.nan)
        school_history_std = school_record.get("simpangan_baku_rapor_sekolah_tahun_lalu", np.nan)
        school_history_acceptance = school_record.get("proporsi_diterima_sekolah_tahun_lalu", np.nan)
        school_history_available = school_record.get("data_sekolah_tahun_lalu_tersedia", 0)
    else:
        school_history_mean = np.nan
        school_history_std = np.nan
        school_history_acceptance = np.nan
        school_history_available = 0

    row = {
        **{f"nilai_rapor_semester_{i + 1}": float(scores[i]) for i in range(5)},
        "rata_rata_rapor": average,
        "tren_rapor": trend,
        "peminatan": specialization,
        "jumlah_siswa_eligible": int(eligible_total),
        "posisi_relatif_eligible": float(eligible_rank) / float(eligible_total),
        "skor_z_rapor_dalam_sekolah": z_score,
        "jumlah_pemilih_prodi_sama": same_program_competitors,
        "jumlah_pemilih_ptn_sama": same_university_competitors,
        "peringkat_di_antara_pemilih_prodi_sama": rank_among_same_program,
        "daya_tampung_tahun_ini": program_record["daya_tampung_tahun_ini"],
        "daya_tampung_tahun_lalu": program_record["daya_tampung_tahun_lalu"],
        "keketatan_tahun_lalu": program_record["keketatan_tahun_lalu"],
        "tren_keketatan": program_record["tren_keketatan"],
        "kelompok_prodi": program_record["kelompok_prodi"],
        "jenjang_prodi": program_record["jenjang_prodi"],
        "akreditasi_prodi": program_record["akreditasi_prodi"],
        "wajib_portofolio": program_record["wajib_portofolio"],
        "prodi_baru": program_record["prodi_baru"],
        "nama_ptn": program_record["nama_ptn"],
        "provinsi_ptn": program_record["provinsi_ptn"],
        "kelinieran_jurusan": int(linearity),
        "kesesuaian_peminatan_kelompok_prodi": match,
        "ptn_satu_provinsi_dengan_sekolah": same_province,
        "akreditasi_sekolah": school_record["akreditasi_sekolah"],
        "jenis_sekolah": school_record["jenis_sekolah"],
        "provinsi_sekolah": school_record["provinsi_sekolah"],
        "persentil_prestasi_sekolah": school_record["persentil_prestasi_sekolah"],
        "rata_rata_rapor_sekolah_tahun_lalu": school_history_mean,
        "simpangan_baku_rapor_sekolah_tahun_lalu": school_history_std,
        "proporsi_diterima_sekolah_tahun_lalu": school_history_acceptance,
        "data_sekolah_tahun_lalu_tersedia": int(school_history_available) if pd.notna(school_history_available) else 0,
        "urutan_pilihan": int(choice_order),
    }
    return row


def get_positive_probability(model, input_data):

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model tidak memiliki method predict_proba()."
        )

    expected = getattr(model, "feature_names_in_", None)

    if expected is None:
        st.error(
            "Model tidak menyediakan feature_names_in_. "
            "Perlu memeriksa pipeline atau booster XGBoost."
        )

        if hasattr(model, "named_steps"):
            st.write("Tahapan pipeline:", list(model.named_steps.keys()))

            final_estimator = model.steps[-1][1]

            if hasattr(final_estimator, "get_booster"):
                st.write(
                    "Nama fitur booster:",
                    final_estimator.get_booster().feature_names
                )

        st.stop()

    expected = list(expected)
    actual = list(input_data.columns)

    missing = [f for f in expected if f not in actual]
    extra = [f for f in actual if f not in expected]

    if missing or extra:
        st.error("Fitur model dan aplikasi tidak cocok.")

        st.write("Jumlah fitur model:", len(expected))
        st.write("Jumlah fitur aplikasi:", len(actual))

        st.write("**Daftar fitur model:**")
        st.code("\n".join(expected))

        st.write("**Fitur yang belum tersedia di aplikasi:**")
        st.code("\n".join(missing) if missing else "Tidak ada")

        st.write("**Fitur tambahan di aplikasi:**")
        st.code("\n".join(extra) if extra else "Tidak ada")

        st.stop()

    input_data = input_data[expected]

    probabilities = model.predict_proba(input_data)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Kelas positif 1 tidak ditemukan.")

    return probabilities[:, classes.index(1)]


# ============================================================
# 3. LOAD FILE DAN TAMPILAN AWAL
# ============================================================

st.title("🎓 Prediksi Peluang Penerimaan SNBP")
st.write(
    "Masukkan informasi akademik, pilih sekolah, lalu tentukan dua program studi. "
    "Aplikasi akan membentuk fitur dan menghitung probabilitas untuk setiap pilihan."
)

missing_files = [
    file.name for file in [MODEL_FILE, DATASET_FILE, MASTER_FILE] if not file.exists()
]
if missing_files:
    st.error(
        "File berikut belum ditemukan di folder yang sama dengan app.py: "
        + ", ".join(missing_files)
    )
    st.stop()

try:
    model = load_model(str(MODEL_FILE))
    if not hasattr(model, "named_steps"):
        raise TypeError(
            "xgboost.pkl harus berisi pipeline lengkap (preprocessor + sampler + model), "
            "bukan XGBClassifier mentah. Simpan objek pipeline hasil fit dari training."
        )
    dataset = load_dataset(str(DATASET_FILE))
    master_raw = load_master(str(MASTER_FILE), MASTER_SHEET)
    master = prepare_master(master_raw)
    school_reference = build_school_reference(dataset)
except Exception as error:
    st.error(f"Gagal memuat model atau data: {error}")
    st.stop()

# Pastikan model menggunakan 37 fitur yang diharapkan.
model_feature_names = getattr(model, "feature_names_in_", None)
if model_feature_names is not None:
    model_feature_names = list(model_feature_names)
    if set(model_feature_names) != set(FEATURE_COLUMNS):
        st.error(
            "Fitur pada xgboost.pkl tidak sama dengan daftar 37 fitur aplikasi. "
            f"Jumlah fitur model: {len(model_feature_names)}; "
            f"jumlah fitur aplikasi: {len(FEATURE_COLUMNS)}. "
            "Pastikan xgboost.pkl berasal dari pipeline training yang sama."
        )
        st.stop()

# Tahun 2027 memakai master 2026 sebagai data terbaru yang tersedia.
target_year = st.selectbox(
    "Tahun SNBP yang diprediksi",
    options=[2027, 2026],
    index=0,
    help="Untuk 2027, master 2026 digunakan sebagai data daya tampung terbaru yang tersedia.",
)

try:
    program_reference, program_metadata = make_program_reference(master, int(target_year))
except Exception as error:
    st.error(f"Gagal menyiapkan referensi program studi: {error}")
    st.stop()

school_codes = sorted(school_reference["kode_sekolah"].dropna().astype(str).unique())
if not school_codes:
    st.error("Tidak ditemukan kode sekolah dalam dataset.")
    st.stop()

st.info(
    "Dataset sudah dipseudonimkan. Karena itu, pilihan sekolah ditampilkan menggunakan "
    "kode sekolah (misalnya LPG_01), bukan nama sekolah asli."
)

st.caption(
    f"Data prodi menggunakan daya tampung tahun {program_metadata['capacity_year']} "
    f"dan keketatan tahun {program_metadata['keketatan_year']}. "
    "Untuk target 2027, program studi baru yang belum ada di master 2026 belum dapat dipilih."
)


# ============================================================
# 4. FORM INPUT SISWA
# ============================================================

with st.form("form_prediksi_snbp"):
    st.header("1. Informasi sekolah")
    school_code = st.selectbox("Pilih kode sekolah", options=school_codes)

    school_record, school_reference_year = get_school_record(
        school_reference, school_code, int(target_year)
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Provinsi", str(school_record["provinsi_sekolah"]))
    col2.metric("Jenis sekolah", str(school_record["jenis_sekolah"]))
    col3.metric("Akreditasi", str(school_record["akreditasi_sekolah"]))
    st.caption(
        f"Profil sekolah diambil dari angkatan referensi {school_reference_year}. "
        "Kode sekolah merupakan kode samaran pada dataset."
    )

    st.header("2. Informasi akademik")
    col1, col2 = st.columns(2)
    with col1:
        eligible_default = school_record.get("jumlah_siswa_eligible", 100)
        if pd.isna(eligible_default) or int(eligible_default) < 1:
            eligible_default = 100
        eligible_total = st.number_input(
            "Jumlah siswa eligible pada tahun target",
            min_value=1,
            max_value=10000,
            value=int(eligible_default),
            step=1,
            help="Nilai awal berasal dari angkatan referensi. Ubah sesuai jumlah eligible tahun target.",
        )
    with col2:
        eligible_rank = st.number_input(
            "Peringkat eligible siswa",
            min_value=1,
            max_value=int(eligible_total),
            value=1,
            step=1,
            help="Peringkat 1 adalah peringkat tertinggi.",
        )

    specialization_options = sorted(
        dataset["peminatan"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )
    preferred_order = [value for value in ["IPA", "IPS", "ALL"] if value in specialization_options]
    specialization_options = preferred_order + [
        value for value in specialization_options if value not in preferred_order
    ]
    if not specialization_options:
        specialization_options = ["IPA", "IPS", "ALL"]
    specialization = st.selectbox(
        "Peminatan siswa",
        options=specialization_options,
        help="ALL digunakan untuk sekolah yang tidak menerapkan penjurusan.",
    )

    st.subheader("Nilai rapor semester 1–5")
    grade_cols = st.columns(5)
    scores = []
    for semester, column in enumerate(grade_cols, start=1):
        with column:
            scores.append(
                st.number_input(
                    f"Semester {semester}",
                    min_value=0.0,
                    max_value=100.0,
                    value=80.0,
                    step=0.1,
                    key=f"grade_semester_{semester}",
                )
            )

    st.divider()
    st.header("3. Pilihan program studi")

    def program_selector(choice_number):
        st.subheader(f"Pilihan {choice_number}")
        universities = sorted(program_reference["nama_ptn"].dropna().unique().tolist())
        selected_university = st.selectbox(
            f"PTN pilihan {choice_number}",
            options=universities,
            key=f"university_{choice_number}",
        )
        candidates = program_reference[
            program_reference["nama_ptn"] == selected_university
        ].sort_values(["nama_prodi", "jenjang_prodi"])
        candidates = candidates.drop_duplicates("kode_prodi")
        program_codes = candidates["kode_prodi"].tolist()
        selected_code = st.selectbox(
            f"Program studi pilihan {choice_number}",
            options=program_codes,
            format_func=lambda code: (
                f"{candidates.loc[candidates['kode_prodi'] == code, 'nama_prodi'].iloc[0]} "
                f"({candidates.loc[candidates['kode_prodi'] == code, 'jenjang_prodi'].iloc[0]})"
            ),
            key=f"program_{choice_number}",
        )
        linearity_label = st.selectbox(
            f"Kelinearan pilihan {choice_number} dengan peminatan",
            options=["LINIER", "TIDAK LINIER"],
            key=f"linearity_{choice_number}",
            help="Pilih sesuai penilaian kelinearan yang digunakan pada data sekolah.",
        )
        selected_program = candidates[candidates["kode_prodi"] == selected_code].iloc[0]
        st.caption(
            f"Kelompok: {selected_program['kelompok_prodi']} · "
            f"Daya tampung acuan: {selected_program['daya_tampung_tahun_ini']} · "
            f"Akreditasi: {selected_program['akreditasi_prodi']}"
        )
        return selected_program, 1 if linearity_label == "LINIER" else 0

    choice_1, linearity_1 = program_selector(1)
    st.divider()
    choice_2, linearity_2 = program_selector(2)

    submitted = st.form_submit_button(
        "🔍 Hitung Prediksi", use_container_width=True
    )


# ============================================================
# 5. MEMBENTUK FITUR DAN MELAKUKAN PREDIKSI
# ============================================================

if submitted:
    if int(eligible_rank) > int(eligible_total):
        st.error("Peringkat eligible tidak boleh lebih besar dari jumlah siswa eligible.")
        st.stop()

    if int(choice_1["kode_prodi"]) == int(choice_2["kode_prodi"]):
        st.error("Pilihan 1 dan pilihan 2 tidak boleh menggunakan program studi yang sama.")
        st.stop()

    if not np.isfinite(np.asarray(scores, dtype=float)).all():
        st.error("Nilai rapor harus berupa angka yang valid.")
        st.stop()

    try:
        rows = []
        for order, (program, linearity) in enumerate(
            [(choice_1, linearity_1), (choice_2, linearity_2)], start=1
        ):
            rows.append(
                build_choice_row(
                    scores=scores,
                    specialization=specialization,
                    eligible_rank=int(eligible_rank),
                    eligible_total=int(eligible_total),
                    school_record=school_record,
                    program_record=program,
                    linearity=linearity,
                    choice_order=order,
                    target_year=int(target_year),
                    school_reference_year=int(school_reference_year),
                )
            )

        input_data = pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS)

        # Validasi akhir: seluruh 37 fitur harus terbentuk.
        if input_data.shape[1] != 37 or list(input_data.columns) != FEATURE_COLUMNS:
            raise ValueError("Struktur fitur input tidak sesuai dengan 37 fitur model.")

        probabilities = get_positive_probability(model, input_data)

        st.success("Prediksi berhasil dihitung.")
        st.header("4. Hasil prediksi")
        result_cols = st.columns(2)
        choices = [choice_1, choice_2]
        for index, (column, program, probability) in enumerate(
            zip(result_cols, choices, probabilities), start=1
        ):
            with column:
                st.subheader(f"Pilihan {index}")
                st.markdown(f"**{program['nama_prodi']}**")
                st.write(program["nama_ptn"])
                st.metric("Probabilitas prediksi diterima", f"{probability * 100:.2f}%")
                st.progress(float(np.clip(probability, 0, 1)))
                st.caption(
                    f"{program['jenjang_prodi']} · {program['kelompok_prodi']} · "
                    f"Provinsi {program['provinsi_ptn']}"
                )

        st.warning(
            "Interpretasi fitur yang belum tersedia: skor_z_rapor_dalam_sekolah, "
            "jumlah_pemilih_prodi_sama, jumlah_pemilih_ptn_sama, dan "
            "peringkat_di_antara_pemilih_prodi_sama tidak dapat dihitung secara tepat "
            "hanya dari dataset_pasangan.parquet. Nilainya dikirim sebagai missing value "
            "dan ditangani oleh imputer di dalam pipeline model. Untuk mengisinya secara "
            "tepat dibutuhkan tabel seluruh siswa eligible beserta seluruh pilihan mereka "
            "pada angkatan target."
        )

        if int(target_year) == 2027:
            st.info(
                "Untuk target 2027, master 2026 digunakan sebagai data daya tampung terbaru. "
                "Fitur statistik sekolah tahun sebelumnya ditandai tidak tersedia karena "
                "dataset yang disediakan belum memuat rekap lengkap angkatan 2026."
            )

        with st.expander("Lihat 37 fitur yang dikirim ke model"):
            display_data = input_data.copy().T
            display_data.columns = ["Pilihan 1", "Pilihan 2"]
            st.dataframe(display_data, use_container_width=True)

        st.caption(
            "Probabilitas merupakan estimasi model berdasarkan pola data historis, "
            "bukan jaminan diterima SNBP."
        )

    except Exception as error:
        st.error(f"Terjadi kesalahan saat prediksi: {error}")
        st.exception(error)
