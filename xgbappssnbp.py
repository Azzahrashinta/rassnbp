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
        "nama_siswa",
        "kode_sekolah",
        "tahun_angkatan",
        "peran_data",
        "diterima",
        "jumlah_siswa_eligible",
        "peminatan",
        "kode_prodi",
        "ptn_pilihan",
        "peringkat_eligible",
        "rata_rata_rapor",
        *SCHOOL_COLUMNS,
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Kolom dataset tidak ditemukan: {sorted(missing)}"
        )
    return data


@st.cache_data
def load_school_mapping(path_string):
    mapping = pd.read_excel(path_string)
    mapping.columns = mapping.columns.astype(str).str.strip()

    required = {"nama_sekolah", "kode_sekolah"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(
            f"Kolom pemetaan sekolah tidak ditemukan: {sorted(missing)}"
        )

    mapping = mapping.dropna(
        subset=["nama_sekolah", "kode_sekolah"]
    ).copy()

    mapping["nama_sekolah"] = (
        mapping["nama_sekolah"].astype(str).str.strip()
    )
    mapping["kode_sekolah"] = (
        mapping["kode_sekolah"].astype(str).str.strip()
    )

    mapping = mapping.drop_duplicates()

    conflicts = (
        mapping.groupby("nama_sekolah")["kode_sekolah"].nunique()
    )
    conflicts = conflicts[conflicts > 1]

    if not conflicts.empty:
        raise ValueError(
            "Ada nama sekolah yang terhubung ke lebih dari satu kode: "
            + ", ".join(conflicts.index.tolist())
        )

    return (
        mapping.sort_values("nama_sekolah")
        .reset_index(drop=True)
    )


@st.cache_data
def load_master(path_string, sheet_name):
    data = pd.read_excel(path_string, sheet_name=sheet_name)
    data.columns = data.columns.astype(str).str.strip()

    required = {
        "Kode Prodi",
        "Universitas",
        "Prodi/Jurusan",
        "Kelompok",
        "Jenjang",
        "Daya Tampung",
        "Keketatan",
        "Portofolio",
        "Akreditasi",
        "Provinsi",
        "Tahun",
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Kolom master SNBP tidak ditemukan: {sorted(missing)}"
        )

    return data


def parse_percentage(value):
    """Mengubah teks seperti '12.47%' menjadi 0.1247."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return number / 100 if number > 1 else number

    text_value = (
        str(value)
        .strip()
        .replace("%", "")
        .replace(",", ".")
    )

    try:
        return float(text_value) / 100
    except ValueError:
        return np.nan


def clean_text(value):
    """Membersihkan teks: trim dan huruf kapital."""
    if pd.isna(value):
        return np.nan
    return str(value).strip().upper()


def prepare_master(master_raw):
    """Menyiapkan kolom master sesuai aturan feature engineering."""
    master = master_raw.copy()

    master["Tahun"] = pd.to_numeric(
        master["Tahun"], errors="coerce"
    ).astype("Int64")

    master["Kode Prodi"] = pd.to_numeric(
        master["Kode Prodi"], errors="coerce"
    )

    master["Daya Tampung"] = pd.to_numeric(
        master["Daya Tampung"], errors="coerce"
    )

    master["keketatan_decimal"] = (
        master["Keketatan"].map(parse_percentage)
    )

    text_columns = [
        "Universitas",
        "Prodi/Jurusan",
        "Kelompok",
        "Jenjang",
        "Akreditasi",
        "Provinsi",
        "Portofolio",
    ]

    for column in text_columns:
        master[column] = master[column].map(clean_text)

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
        master["Akreditasi"]
        .map(accreditation_map)
        .fillna("BELUM ATAU SEMENTARA")
    )

    master["wajib_portofolio_model"] = np.where(
        master["Portofolio"].isna(),
        np.nan,
        (master["Portofolio"] != "TIDAK ADA").astype(int),
    )

    return master


def make_program_reference(master, target_year):
    """
    Membentuk referensi prodi dari gabungan seluruh tahun <= target_year.

    Data historis tetap berada di master. Referensi prediksi berisi satu
    baris per kode prodi, dengan metadata terbaru dan fitur historis.
    """
    data = master.copy()

    data["Tahun"] = pd.to_numeric(
        data["Tahun"], errors="coerce"
    )
    data["Kode Prodi"] = pd.to_numeric(
        data["Kode Prodi"], errors="coerce"
    )
    data["Daya Tampung"] = pd.to_numeric(
        data["Daya Tampung"], errors="coerce"
    )

    data = data.dropna(
        subset=["Tahun", "Kode Prodi"]
    ).copy()

    data = data[
        data["Tahun"] <= int(target_year)
    ].copy()

    if data.empty:
        raise ValueError(
            "Tidak ada data master pada tahun target atau sebelumnya."
        )

    # Satu catatan per kode prodi per tahun.
    data = data.sort_values(
        ["Kode Prodi", "Tahun"]
    )
    data = data.drop_duplicates(
        ["Kode Prodi", "Tahun"],
        keep="last"
    )

    # Metadata terbaru per kode prodi.
    latest_metadata = data.drop_duplicates(
        "Kode Prodi",
        keep="last"
    ).copy()

    latest_metadata = latest_metadata.dropna(
        subset=["Universitas", "Prodi/Jurusan"]
    )

    # Daya tampung terbaru yang tersedia.
    capacity = data.dropna(
        subset=["Daya Tampung"]
    ).copy()

    latest_capacity = (
        capacity.drop_duplicates("Kode Prodi", keep="last")[
            ["Kode Prodi", "Tahun", "Daya Tampung"]
        ]
        .rename(
            columns={
                "Kode Prodi": "kode_prodi",
                "Tahun": "tahun_daya_tampung",
                "Daya Tampung": "daya_tampung_tahun_ini",
            }
        )
    )

    # Daya tampung tahun sebelum target.
    previous_capacity = (
        data[data["Tahun"] == int(target_year) - 1][
            ["Kode Prodi", "Daya Tampung"]
        ]
        .rename(
            columns={
                "Kode Prodi": "kode_prodi",
                "Daya Tampung": "daya_tampung_tahun_lalu",
            }
        )
    )

    # Dua nilai keketatan terbaru per prodi.
    comp = data.dropna(
        subset=["keketatan_decimal"]
    )[
        ["Kode Prodi", "Tahun", "keketatan_decimal"]
    ].sort_values(
        ["Kode Prodi", "Tahun"],
        ascending=[True, False]
    )

    comp = comp.drop_duplicates(
        ["Kode Prodi", "Tahun"],
        keep="last"
    )

    comp["urutan_terbaru"] = (
        comp.groupby("Kode Prodi").cumcount()
    )

    latest_comp = (
        comp[comp["urutan_terbaru"] == 0][
            ["Kode Prodi", "keketatan_decimal"]
        ]
        .rename(
            columns={
                "Kode Prodi": "kode_prodi",
                "keketatan_decimal": "keketatan_tahun_lalu",
            }
        )
    )

    previous_comp = (
        comp[comp["urutan_terbaru"] == 1][
            ["Kode Prodi", "keketatan_decimal"]
        ]
        .rename(
            columns={
                "Kode Prodi": "kode_prodi",
                "keketatan_decimal": "keketatan_sebelumnya",
            }
        )
    )

    program = latest_metadata[
        [
            "Kode Prodi",
            "Universitas",
            "Prodi/Jurusan",
            "Kelompok",
            "Jenjang",
            "akreditasi_model",
            "wajib_portofolio_model",
            "Provinsi",
        ]
    ].rename(
        columns={
            "Kode Prodi": "kode_prodi",
            "Universitas": "nama_ptn",
            "Prodi/Jurusan": "nama_prodi",
            "Kelompok": "kelompok_prodi",
            "Jenjang": "jenjang_prodi",
            "akreditasi_model": "akreditasi_prodi",
            "wajib_portofolio_model": "wajib_portofolio",
            "Provinsi": "provinsi_ptn",
        }
    )

    for extra in [
        latest_capacity,
        previous_capacity,
        latest_comp,
        previous_comp,
    ]:
        program = program.merge(
            extra,
            on="kode_prodi",
            how="left",
            validate="one_to_one",
        )

    program["tren_keketatan"] = (
        program["keketatan_tahun_lalu"]
        - program["keketatan_sebelumnya"]
    )

    program["prodi_baru"] = (
        program["daya_tampung_tahun_lalu"]
        .isna()
        .astype(int)
    )

    program = program.drop(
        columns=["keketatan_sebelumnya"],
        errors="ignore"
    )

    program["kode_prodi"] = pd.to_numeric(
        program["kode_prodi"], errors="coerce"
    )

    program = program.dropna(
        subset=["nama_ptn", "nama_prodi", "kode_prodi"]
    )

    # Normalisasi nama untuk menghindari masalah spasi/kapitalisasi.
    for column in [
        "nama_ptn",
        "nama_prodi",
        "kelompok_prodi",
        "jenjang_prodi",
        "akreditasi_prodi",
        "provinsi_ptn",
    ]:
        program[column] = program[column].map(clean_text)

    program = program.sort_values(
        ["nama_ptn", "nama_prodi", "jenjang_prodi", "kode_prodi"]
    )

    program = program.drop_duplicates(
        "kode_prodi",
        keep="first"
    ).reset_index(drop=True)

    years = sorted(
        data["Tahun"].dropna().astype(int).unique().tolist()
    )

    metadata = {
        "jumlah_program_studi": int(
            program["kode_prodi"].nunique()
        ),
        "jumlah_universitas": int(
            program["nama_ptn"].nunique()
        ),
        "tahun_metadata": years,
        "capacity_year": (
            int(capacity["Tahun"].max())
            if not capacity.empty
            else None
        ),
        "previous_capacity_year": int(target_year) - 1,
        "keketatan_year": (
            int(comp["Tahun"].max())
            if not comp.empty
            else None
        ),
    }

    return program, metadata


def build_school_reference(data):
    """Satu baris referensi per sekolah-angkatan.

    Kolom profil sekolah diasumsikan konstan pada setiap kelompok
    sekolah-angkatan. Drop_duplicates menghindari perilaku groupby.first()
    yang dapat mengambil nilai non-null dari baris berbeda.
    """
    required = ["kode_sekolah", "tahun_angkatan"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Kolom referensi sekolah tidak tersedia: {sorted(missing)}")

    temp = data.copy()
    temp["kode_sekolah"] = temp["kode_sekolah"].astype(str).str.strip()
    temp["tahun_angkatan"] = pd.to_numeric(
        temp["tahun_angkatan"], errors="coerce"
    )
    temp = temp.dropna(subset=["kode_sekolah", "tahun_angkatan"])
    temp["tahun_angkatan"] = temp["tahun_angkatan"].astype(int)

    return (
        temp.sort_values(["kode_sekolah", "tahun_angkatan"])
        .drop_duplicates(["kode_sekolah", "tahun_angkatan"], keep="first")
        .reset_index(drop=True)
    )


def get_school_record(school_ref, school_code, target_year):
    """Mengambil angkatan sekolah terakhir yang lebih lama dari tahun target.

    Penggunaan tahun yang lebih lama mencegah kebocoran informasi dari
    angkatan yang sedang diprediksi.
    """
    records = school_ref[
        school_ref["kode_sekolah"].astype(str).str.strip()
        == str(school_code).strip()
    ].copy()

    if records.empty:
        raise ValueError("Informasi sekolah tidak ditemukan di dataset.")

    records["tahun_angkatan"] = pd.to_numeric(
        records["tahun_angkatan"], errors="coerce"
    )
    past_records = records[
        records["tahun_angkatan"] < int(target_year)
    ].copy()

    if past_records.empty:
        raise ValueError(
            f"Tidak tersedia data angkatan sebelum {target_year} untuk sekolah ini."
        )

    reference_year = int(past_records["tahun_angkatan"].max())
    row = past_records.loc[
        past_records["tahun_angkatan"] == reference_year
    ].iloc[0].copy()

    return row, reference_year


def get_reference_cohort(data, school_code, target_year):
    """Mengambil seluruh pasangan pilihan dari angkatan sekolah sebelumnya.

    Angkatan referensi adalah angkatan terbaru yang lebih kecil dari tahun
    target. Dataset berisi beberapa baris per siswa karena satu siswa dapat
    memiliki lebih dari satu pilihan prodi.
    """
    temp = data.copy()
    temp["kode_sekolah"] = temp["kode_sekolah"].astype(str).str.strip()
    temp["tahun_angkatan"] = pd.to_numeric(
        temp["tahun_angkatan"], errors="coerce"
    )

    school_rows = temp[
        (temp["kode_sekolah"] == str(school_code).strip())
        & (temp["tahun_angkatan"] < int(target_year))
    ].copy()

    if school_rows.empty:
        raise ValueError(
            f"Tidak ada data pasangan siswa untuk sekolah ini sebelum {target_year}."
        )

    reference_year = int(school_rows["tahun_angkatan"].max())
    cohort = school_rows[
        school_rows["tahun_angkatan"] == reference_year
    ].copy()

    cohort["nama_siswa"] = cohort["nama_siswa"].astype(str).str.strip()
    cohort["kode_prodi_num"] = pd.to_numeric(
        cohort["kode_prodi"], errors="coerce"
    )
    cohort["peringkat_eligible_num"] = pd.to_numeric(
        cohort["peringkat_eligible"], errors="coerce"
    )
    cohort["rata_rata_rapor_num"] = pd.to_numeric(
        cohort["rata_rata_rapor"], errors="coerce"
    )
    cohort["ptn_pilihan_norm"] = cohort["ptn_pilihan"].map(clean_text)

    return cohort, reference_year


def compute_competition_features(
    cohort,
    average_score,
    eligible_rank,
    program_code,
    university_name,
):
    """Menghitung z-score dan fitur persaingan berbasis angkatan referensi.

    Setiap hitungan menggunakan siswa unik, bukan jumlah baris pasangan.
    Satu siswa yang memiliki dua pilihan tetap dihitung satu kali untuk
    jumlah pemilih prodi/PTN yang sama.
    """
    if cohort is None or cohort.empty:
        return np.nan, np.nan, np.nan, np.nan

    # Satu observasi nilai per siswa untuk distribusi nilai sekolah.
    student_scores = (
        cohort[["nama_siswa", "rata_rata_rapor_num"]]
        .dropna(subset=["nama_siswa", "rata_rata_rapor_num"])
        .drop_duplicates("nama_siswa", keep="first")
    )

    if len(student_scores) >= 2:
        cohort_mean = float(student_scores["rata_rata_rapor_num"].mean())
        # ddof=1 mengikuti perilaku default pandas Series.std().
        cohort_std = float(student_scores["rata_rata_rapor_num"].std(ddof=1))
        z_score = (
            (float(average_score) - cohort_mean) / cohort_std
            if np.isfinite(cohort_std) and cohort_std > 0
            else np.nan
        )
    else:
        z_score = np.nan

    program_code_num = pd.to_numeric(program_code, errors="coerce")
    same_program = cohort[
        cohort["kode_prodi_num"] == program_code_num
    ].copy()
    same_program_students = same_program["nama_siswa"].nunique()

    university_norm = clean_text(university_name)
    same_university = cohort[
        cohort["ptn_pilihan_norm"] == university_norm
    ].copy()
    same_university_students = same_university["nama_siswa"].nunique()

    # Tambahkan satu siswa yang sedang diprediksi ke hitungan historis.
    same_program_count = int(same_program_students + 1)
    same_university_count = int(same_university_students + 1)

    # Peringkat kompetisi: 1 + jumlah pemilih prodi yang peringkat
    # eligible-nya lebih baik (angka peringkat lebih kecil).
    competitor_ranks = (
        same_program[["nama_siswa", "peringkat_eligible_num"]]
        .dropna(subset=["peringkat_eligible_num"])
        .drop_duplicates("nama_siswa", keep="first")
    )
    better_competitors = int(
        (competitor_ranks["peringkat_eligible_num"] < int(eligible_rank)).sum()
    )
    rank_among_program = better_competitors + 1

    return (
        float(z_score) if pd.notna(z_score) else np.nan,
        same_program_count,
        same_university_count,
        rank_among_program,
    )

def compute_report_features(scores):
    """Menghitung rata-rata dan slope tren rapor."""
    semester_numbers = np.arange(1, 6, dtype=float)
    values = np.array(scores, dtype=float)
    mask = np.isfinite(values)

    average = (
        float(np.mean(values[mask]))
        if mask.any()
        else np.nan
    )

    if mask.sum() >= 2:
        trend = float(
            np.polyfit(
                semester_numbers[mask],
                values[mask],
                1
            )[0]
        )
    else:
        trend = np.nan

    return average, trend


def compute_match_features(
    specialization,
    program_group,
    school_province,
    ptn_province
):
    """Menghitung kesesuaian peminatan dan kesamaan provinsi."""
    if pd.isna(program_group):
        match = np.nan
    elif specialization == "ALL" or program_group == "VOKASI":
        match = "TIDAK BERLAKU"
    else:
        pairs = {
            ("IPA", "SAINTEK"),
            ("IPS", "SOSHUM"),
        }
        match = (
            "SESUAI"
            if (specialization, program_group) in pairs
            else "TIDAK SESUAI"
        )

    if pd.isna(ptn_province):
        same_province = np.nan
    else:
        provinces = [
            part.strip().upper()
            for part in str(ptn_province).split(";")
        ]
        same_province = int(
            str(school_province).strip().upper() in provinces
        )

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
    reference_cohort,
):
    average, trend = compute_report_features(scores)

    match, same_province = compute_match_features(
        specialization,
        program_record["kelompok_prodi"],
        school_record["provinsi_sekolah"],
        program_record["provinsi_ptn"],
    )

    (
        z_score,
        same_program_competitors,
        same_university_competitors,
        rank_among_same_program,
    ) = compute_competition_features(
        cohort=reference_cohort,
        average_score=average,
        eligible_rank=eligible_rank,
        program_code=program_record["kode_prodi"],
        university_name=program_record["nama_ptn"],
    )

    # Statistik sekolah pada baris cohort referensi merupakan ringkasan
    # tahun sebelumnya. Gunakan hanya jika penanda ketersediaannya aktif.
    school_history_available = school_record.get(
        "data_sekolah_tahun_lalu_tersedia", 0
    )
    if pd.notna(school_history_available) and int(school_history_available) == 1:
        school_history_mean = school_record.get(
            "rata_rata_rapor_sekolah_tahun_lalu", np.nan
        )
        school_history_std = school_record.get(
            "simpangan_baku_rapor_sekolah_tahun_lalu", np.nan
        )
        school_history_acceptance = school_record.get(
            "proporsi_diterima_sekolah_tahun_lalu", np.nan
        )
    else:
        school_history_mean = np.nan
        school_history_std = np.nan
        school_history_acceptance = np.nan
        school_history_available = 0

    row = {
        **{
            f"nilai_rapor_semester_{i + 1}": float(scores[i])
            for i in range(5)
        },
        "rata_rata_rapor": average,
        "tren_rapor": trend,
        "peminatan": specialization,
        "peringkat_eligible": int(eligible_rank),
        "jumlah_siswa_eligible": int(eligible_total),
        "posisi_relatif_eligible": (
            float(eligible_rank) / float(eligible_total)
        ),
        "skor_z_rapor_dalam_sekolah": z_score,
        "jumlah_pemilih_prodi_sama": same_program_competitors,
        "jumlah_pemilih_ptn_sama": same_university_competitors,
        "peringkat_di_antara_pemilih_prodi_sama": (
            rank_among_same_program
        ),
        "daya_tampung_tahun_ini": program_record[
            "daya_tampung_tahun_ini"
        ],
        "daya_tampung_tahun_lalu": program_record[
            "daya_tampung_tahun_lalu"
        ],
        "keketatan_tahun_lalu": program_record[
            "keketatan_tahun_lalu"
        ],
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
        "persentil_prestasi_sekolah": (
            school_record["persentil_prestasi_sekolah"]
        ),
        "rata_rata_rapor_sekolah_tahun_lalu": (
            school_history_mean
        ),
        "simpangan_baku_rapor_sekolah_tahun_lalu": (
            school_history_std
        ),
        "proporsi_diterima_sekolah_tahun_lalu": (
            school_history_acceptance
        ),
        "data_sekolah_tahun_lalu_tersedia": (
            int(school_history_available)
            if pd.notna(school_history_available)
            else 0
        ),
        "urutan_pilihan": int(choice_order),
    }

    return row


def get_positive_probability(model, input_frame):
    """Mengambil probabilitas kelas 1 dari pipeline/model."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Objek catboost.pkl tidak memiliki method predict_proba()."
        )

    model_features = getattr(
        model, "feature_names_in_", None
    )

    if model_features is not None:
        model_features = list(model_features)

        missing = (
            set(model_features) - set(input_frame.columns)
        )
        extra = (
            set(input_frame.columns) - set(model_features)
        )

        if missing or extra:
            raise ValueError(
                "Fitur input tidak sama dengan fitur model. "
                f"Kurang: {sorted(missing)}; "
                f"berlebih: {sorted(extra)}"
            )

        input_frame = input_frame.loc[:, model_features]
    else:
        input_frame = input_frame.loc[:, FEATURE_COLUMNS]

    probabilities = model.predict_proba(input_frame)

    classes = list(
        getattr(model, "classes_", [0, 1])
    )

    if 1 not in classes:
        raise ValueError(
            "Model tidak memiliki kelas target 1 (diterima)."
        )

    positive_index = classes.index(1)

    return probabilities[:, positive_index]


# ============================================================
# 3. LOAD FILE DAN TAMPILAN AWAL
# ============================================================

st.title("🎓 Prediksi SNBP")

st.write(
    "Masukkan informasi akademik, pilih sekolah, lalu tentukan "
    "dua program studi. Aplikasi akan membentuk fitur dan "
    "menghitung probabilitas untuk setiap pilihan."
)

required_files = [
    MODEL_FILE,
    DATASET_FILE,
    MASTER_FILE,
    SCHOOL_MAPPING_FILE,
]

missing_files = [
    file.name
    for file in required_files
    if not file.exists()
]

if missing_files:
    st.error(
        "File berikut belum ditemukan di folder yang sama dengan app.py: "
        + ", ".join(missing_files)
    )
    st.stop()


try:
    model = load_model(str(MODEL_FILE))

    dataset = load_dataset(str(DATASET_FILE))
    master_raw = load_master(
        str(MASTER_FILE),
        MASTER_SHEET
    )
    master = prepare_master(master_raw)

    school_reference = build_school_reference(dataset)
    school_mapping = load_school_mapping(
        str(SCHOOL_MAPPING_FILE)
    )

except Exception as error:
    st.error(
        f"Gagal memuat model atau data: {error}"
    )
    st.exception(error)
    st.stop()


# Validasi fitur model.
model_feature_names = getattr(
    model, "feature_names_in_", None
)

if model_feature_names is not None:
    model_feature_names = list(model_feature_names)

    if model_feature_names != FEATURE_COLUMNS:
        st.error(
            "Urutan/nama fitur aplikasi belum sama dengan fitur input model. "
            f"Model: {len(model_feature_names)} fitur; "
            f"aplikasi: {len(FEATURE_COLUMNS)} fitur."
        )
        st.write("Fitur model:", model_feature_names)
        st.write("Fitur aplikasi:", FEATURE_COLUMNS)
        st.stop()


# ============================================================
# 4. TAHUN PREDIKSI DAN REFERENSI PROGRAM STUDI
# ============================================================

target_year = st.selectbox(
    "Tahun SNBP yang diprediksi",
    options=[2027, 2026],
    index=0,
    help=(
        "Untuk 2027, master 2026 digunakan sebagai data "
        "daya tampung terbaru yang tersedia."
    ),
    key="target_year"
)

try:
    program_reference, program_metadata = make_program_reference(
        master,
        int(target_year)
    )

except Exception as error:
    st.error(
        f"Gagal menyiapkan referensi program studi: {error}"
    )
    st.exception(error)
    st.stop()


st.caption(
    f"Daftar pilihan menggabungkan data master tahun "
    f"{', '.join(map(str, program_metadata['tahun_metadata']))}. "
    f"Tersedia {program_metadata['jumlah_program_studi']:,} "
    f"program studi dari "
    f"{program_metadata['jumlah_universitas']:,} universitas. "
    "Metadata menggunakan catatan terbaru yang tersedia "
    "untuk setiap program studi."
)


# ============================================================
# 5. DIAGNOSTIK DATA MASTER (DAPAT DIBUKA/TUTUP)
# ============================================================

with st.expander(
    "🔎 Diagnostik Data Master",
    expanded=False
):
    st.write("### Data Excel")

    st.write({
        "Jumlah baris master": len(master),
        "Jumlah kode prodi unik": master["Kode Prodi"].nunique(),
        "Tahun tersedia": sorted(
            master["Tahun"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
    })

    st.write("### Referensi program studi")

    st.write({
        "Jumlah baris referensi": len(program_reference),
        "Jumlah kode prodi unik": (
            program_reference["kode_prodi"].nunique()
        ),
        "Jumlah universitas": (
            program_reference["nama_ptn"].nunique()
        ),
    })

    university_counts = (
        program_reference
        .groupby("nama_ptn")
        .agg(
            jumlah_prodi=("kode_prodi", "nunique")
        )
        .sort_values("jumlah_prodi", ascending=False)
        .reset_index()
    )

    st.write("### Jumlah prodi per universitas")

    st.dataframe(
        university_counts,
        use_container_width=True,
        hide_index=True
    )

    st.write("### Data historis per tahun")

    yearly_counts = (
        master.groupby("Tahun")
        .agg(
            jumlah_baris=("Kode Prodi", "size"),
            jumlah_prodi=("Kode Prodi", "nunique")
        )
        .reset_index()
    )

    st.dataframe(
        yearly_counts,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 6. INPUT SISWA
# ============================================================

st.header("1. Informasi sekolah")

school_names = (
    school_mapping["nama_sekolah"]
    .dropna()
    .astype(str)
    .sort_values()
    .unique()
    .tolist()
)

if not school_names:
    st.error(
        "Tidak ada nama sekolah yang tersedia pada mapping_sekolah.xlsx."
    )
    st.stop()

selected_school_name = st.selectbox(
    "Nama sekolah",
    options=school_names,
    help=(
        "Ketik nama sekolah untuk mencari. "
        "Profil sekolah akan dihubungkan otomatis."
    ),
    key="selected_school"
)

school_code = school_mapping.loc[
    school_mapping["nama_sekolah"] == selected_school_name,
    "kode_sekolah"
].iloc[0]

try:
    school_record, school_reference_year = get_school_record(
        school_reference,
        school_code,
        int(target_year)
    )
    reference_cohort, competition_reference_year = get_reference_cohort(
        dataset,
        school_code,
        int(target_year)
    )
    if competition_reference_year != school_reference_year:
        raise ValueError(
            "Tahun referensi statistik sekolah dan persaingan tidak sama. "
            f"Sekolah={school_reference_year}, persaingan={competition_reference_year}."
        )
except Exception as error:
    st.error(
        f"Gagal mengambil informasi sekolah: {error}"
    )
    st.stop()


# ============================================================
# 7. INFORMASI AKADEMIK
# ============================================================

st.header("2. Informasi akademik")

col1, col2 = st.columns(2)

with col1:
    eligible_default = school_record.get(
        "jumlah_siswa_eligible",
        100
    )

    if (
        pd.isna(eligible_default)
        or int(eligible_default) < 1
    ):
        eligible_default = 100

    eligible_total = st.number_input(
        "Jumlah siswa eligible pada tahun target",
        min_value=1,
        max_value=10000,
        value=int(eligible_default),
        step=1,
        help=(
            "Nilai awal berasal dari angkatan referensi. "
            "Ubah sesuai jumlah eligible tahun target."
        ),
        key="eligible_total"
    )

with col2:
    eligible_rank = st.number_input(
        "Peringkat eligible siswa",
        min_value=1,
        max_value=int(eligible_total),
        value=1,
        step=1,
        help="Peringkat 1 adalah peringkat tertinggi.",
        key="eligible_rank"
    )


# Peminatan siswa.
specialization_options = sorted(
    dataset["peminatan"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
    .unique()
    .tolist()
)

preferred_order = [
    value
    for value in ["IPA", "IPS", "ALL"]
    if value in specialization_options
]

specialization_options = preferred_order + [
    value
    for value in specialization_options
    if value not in preferred_order
]

if not specialization_options:
    specialization_options = ["IPA", "IPS", "ALL"]

specialization = st.selectbox(
    "Peminatan siswa",
    options=specialization_options,
    help=(
        "ALL digunakan untuk sekolah yang tidak menerapkan penjurusan."
    ),
    key="specialization"
)


# Nilai rapor.
st.subheader("Nilai rapor semester 1–5")

grade_cols = st.columns(5)
scores = []

for semester, column in enumerate(
    grade_cols,
    start=1
):
    with column:
        score = st.number_input(
            f"Semester {semester}",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1,
            key=f"grade_semester_{semester}"
        )
        scores.append(score)


# ============================================================
# 8. PEMILIHAN PROGRAM STUDI
# ============================================================

st.divider()
st.header("3. Pilihan program studi")


def reset_program_selection(choice_number):
    """
    Menghapus pilihan prodi lama saat universitas berubah.
    Callback dijalankan sebelum aplikasi dirender ulang.
    """
    program_key = f"program_{choice_number}"

    if program_key in st.session_state:
        del st.session_state[program_key]


# Pastikan nama universitas bersih dan konsisten.
universities = sorted(
    program_reference["nama_ptn"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)


def program_selector(choice_number):
    st.subheader(f"Pilihan {choice_number}")

    university_key = f"university_{choice_number}"
    program_key = f"program_{choice_number}"
    linearity_key = f"linearity_{choice_number}"

    # Pilihan universitas.
    selected_university = st.selectbox(
        f"PTN pilihan {choice_number}",
        options=universities,
        key=university_key,
        on_change=reset_program_selection,
        args=(choice_number,)
    )

    # Filter prodi berdasarkan universitas yang dipilih.
    candidates = program_reference.loc[
        program_reference["nama_ptn"] == selected_university
    ].copy()

    candidates = candidates.sort_values(
        [
            "nama_prodi",
            "jenjang_prodi",
            "kode_prodi"
        ]
    )

    candidates = candidates.drop_duplicates(
        subset=["kode_prodi"],
        keep="first"
    ).reset_index(drop=True)

    if candidates.empty:
        st.warning(
            f"Tidak ada program studi ditemukan untuk {selected_university}."
        )
        return None, None

    # Jumlah prodi yang tersedia.
    st.caption(
        f"{len(candidates):,} program studi tersedia "
        f"di {selected_university}."
    )

    # Dictionary label prodi.
    program_labels = {
        row["kode_prodi"]: (
            f"{row['nama_prodi']} ({row['jenjang_prodi']})"
        )
        for _, row in candidates.iterrows()
    }

    program_codes = candidates["kode_prodi"].tolist()

    # Jika state pilihan sebelumnya tidak lagi valid,
    # hapus agar Streamlit memilih opsi valid pertama.
    current_program = st.session_state.get(program_key)

    if (
        current_program is not None
        and current_program not in program_codes
    ):
        del st.session_state[program_key]

    selected_code = st.selectbox(
        f"Program studi pilihan {choice_number}",
        options=program_codes,
        format_func=lambda code: program_labels[code],
        key=program_key
    )

    # Kelinearan jurusan.
    linearity_label = st.selectbox(
        f"Kelinearan pilihan {choice_number} dengan peminatan",
        options=["LINIER", "TIDAK LINIER"],
        key=linearity_key,
        help=(
            "Pilih sesuai penilaian kelinearan "
            "yang digunakan pada data sekolah."
        )
    )

    selected_program = candidates.loc[
        candidates["kode_prodi"] == selected_code
    ].iloc[0]

    st.info(
        f"**{selected_program['nama_prodi']}**\n\n"
        f"Universitas: {selected_program['nama_ptn']}\n\n"
        f"Kode prodi: {int(selected_program['kode_prodi'])}\n\n"
        f"Jenjang: {selected_program['jenjang_prodi']}\n\n"
        f"Kelompok: {selected_program['kelompok_prodi']}\n\n"
        f"Daya tampung acuan: "
        f"{selected_program['daya_tampung_tahun_ini']}\n\n"
        f"Akreditasi: {selected_program['akreditasi_prodi']}"
    )

    return (
        selected_program,
        1 if linearity_label == "LINIER" else 0
    )


# Pilihan pertama.
choice_1, linearity_1 = program_selector(1)

st.divider()

# Pilihan kedua.
choice_2, linearity_2 = program_selector(2)


# ============================================================
# 9. TOMBOL PREDIKSI
# ============================================================

st.divider()

submitted = st.button(
    "🔍 Hitung Prediksi",
    use_container_width=True,
    type="primary",
    key="button_predict"
)


# ============================================================
# 10. MEMBENTUK FITUR DAN MELAKUKAN PREDIKSI
# ============================================================

if submitted:

    if choice_1 is None or choice_2 is None:
        st.error(
            "Pastikan kedua pilihan program studi sudah tersedia."
        )
        st.stop()

    if int(eligible_rank) > int(eligible_total):
        st.error(
            "Peringkat eligible tidak boleh lebih besar "
            "dari jumlah siswa eligible."
        )
        st.stop()

    if (
        int(choice_1["kode_prodi"])
        == int(choice_2["kode_prodi"])
    ):
        st.error(
            "Pilihan 1 dan pilihan 2 tidak boleh "
            "menggunakan program studi yang sama."
        )
        st.stop()

    if not np.isfinite(
        np.asarray(scores, dtype=float)
    ).all():
        st.error(
            "Nilai rapor harus berupa angka yang valid."
        )
        st.stop()

    try:
        rows = []

        selected_choices = [
            (choice_1, linearity_1),
            (choice_2, linearity_2),
        ]

        for order, (program, linearity) in enumerate(
            selected_choices,
            start=1
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
                    school_reference_year=int(
                        school_reference_year
                    ),
                    reference_cohort=reference_cohort,
                )
            )

        input_data = (
            pd.DataFrame(rows)
            .reindex(columns=FEATURE_COLUMNS)
        )

        # Validasi struktur 38 fitur.
        if (
            input_data.shape[1] != 38
            or list(input_data.columns) != FEATURE_COLUMNS
        ):
            raise ValueError(
                "Struktur fitur input tidak sesuai "
                "dengan 38 fitur model."
            )

        probabilities = get_positive_probability(
            model,
            input_data
        )

        st.info(
            f"Fitur persaingan dihitung menggunakan data historis "
            f"angkatan {competition_reference_year} dari sekolah yang dipilih. "
            "Angka ini merupakan proksi persaingan berdasarkan angkatan "
            "referensi, bukan jumlah pesaing aktual pada tahun target."
        )

        st.success(
            "Prediksi berhasil dihitung."
        )

        st.header("4. Hasil prediksi")

        result_cols = st.columns(2)
        choices = [choice_1, choice_2]

        for index, (
            column,
            program,
            probability
        ) in enumerate(
            zip(
                result_cols,
                choices,
                probabilities
            ),
            start=1
        ):
            with column:
                st.subheader(f"Pilihan {index}")

                st.markdown(
                    f"**{program['nama_prodi']}**"
                )

                st.write(
                    program["nama_ptn"]
                )

                st.metric(
                    "Probabilitas prediksi diterima",
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(
                        np.clip(
                            probability,
                            0,
                            1
                        )
                    )
                )

                st.caption(
                    f"{program['jenjang_prodi']} · "
                    f"{program['kelompok_prodi']} · "
                    f"Provinsi {program['provinsi_ptn']}"
                )

        st.caption(
            "Probabilitas merupakan estimasi model berdasarkan "
            "pola data historis, bukan jaminan diterima SNBP."
        )

    except Exception as error:
        st.error(
            f"Terjadi kesalahan saat prediksi: {error}"
        )
        st.exception(error)
