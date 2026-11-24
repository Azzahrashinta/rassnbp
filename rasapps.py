
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("xgboost.pkl")

model = load_model()


# ==========================================
# 2. KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="SNBP Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 SNBP Prediction System")
st.write(
    "Masukkan data akademik dan pilihan program studi "
    "untuk memperoleh hasil prediksi SNBP."
)

st.divider()


# ==========================================
# 3. FORM INPUT DATA
# ==========================================

with st.form("form_prediction"):

    # --------------------------------------
    # A. NILAI RAPOR
    # --------------------------------------

    st.subheader("1. Nilai Rapor Semester 1–5")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        semester_1 = st.number_input(
            "Semester 1",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1
        )

    with col2:
        semester_2 = st.number_input(
            "Semester 2",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1
        )

    with col3:
        semester_3 = st.number_input(
            "Semester 3",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1
        )

    with col4:
        semester_4 = st.number_input(
            "Semester 4",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1
        )

    with col5:
        semester_5 = st.number_input(
            "Semester 5",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.1
        )

    st.divider()

    # --------------------------------------
    # B. INFORMASI ELIGIBLE
    # --------------------------------------

    st.subheader("2. Informasi Eligible")

    col1, col2 = st.columns(2)

    with col1:
        peringkat_eligible = st.number_input(
            "Peringkat Eligible",
            min_value=1,
            value=1,
            step=1
        )

    with col2:
        jumlah_eligible = st.number_input(
            "Jumlah Siswa Eligible",
            min_value=1,
            value=100,
            step=1
        )

    st.divider()

    # --------------------------------------
    # C. JURUSAN ASAL SEKOLAH
    # --------------------------------------

    st.subheader("3. Jurusan Asal Sekolah")

    jurusan_sekolah = st.text_input(
        "Jurusan Sekolah",
        placeholder="Contoh: IPA, IPS, atau sesuai kategori data training"
    )

    st.divider()

    # --------------------------------------
    # D. PILIHAN PROGRAM STUDI 1
    # --------------------------------------

    st.subheader("4. Pilihan Program Studi 1")

    col1, col2 = st.columns(2)

    with col1:
        jurusan_1 = st.text_input(
            "Program Studi Pilihan 1",
            placeholder="Contoh: Teknik Informatika"
        )

    with col2:
        universitas_1 = st.text_input(
            "Universitas Pilihan 1",
            placeholder="Contoh: Universitas Indonesia"
        )

    linearitas_1 = st.selectbox(
        "Kelinearan Pilihan 1 dengan Jurusan Sekolah",
        options=[1, 0],
        format_func=lambda x: "Linear" if x == 1 else "Tidak Linear"
    )

    st.divider()

    # --------------------------------------
    # E. PILIHAN PROGRAM STUDI 2
    # --------------------------------------

    st.subheader("5. Pilihan Program Studi 2")

    col1, col2 = st.columns(2)

    with col1:
        jurusan_2 = st.text_input(
            "Program Studi Pilihan 2",
            placeholder="Contoh: Statistika"
        )

    with col2:
        universitas_2 = st.text_input(
            "Universitas Pilihan 2",
            placeholder="Contoh: Universitas Gadjah Mada"
        )

    linearitas_2 = st.selectbox(
        "Kelinearan Pilihan 2 dengan Jurusan Sekolah",
        options=[1, 0],
        format_func=lambda x: "Linear" if x == 1 else "Tidak Linear"
    )

    st.divider()

    # --------------------------------------
    # F. TOMBOL PREDIKSI
    # --------------------------------------

    submit = st.form_submit_button(
        "🔍 Prediksi SNBP",
        use_container_width=True
    )



# ==========================================
# 4. PROSES PREDIKSI
# ==========================================

if submit:

    # Validasi input
    if not all([
        jurusan_sekolah.strip(),
        jurusan_1.strip(),
        universitas_1.strip(),
        jurusan_2.strip(),
        universitas_2.strip()
    ]):
        st.error("Semua informasi jurusan dan universitas harus diisi.")

    elif peringkat_eligible > jumlah_eligible:
        st.error(
            "Peringkat eligible tidak boleh melebihi "
            "jumlah siswa eligible."
        )

    else:

        # Membentuk dataframe input
        input_data = pd.DataFrame([{
            "semester_1": nilai_rapor_semester_1',
            "semester_2": nilai_rapor_semester_2,
            "semester_3": nilai_rapor_semester_3,
            "semester_4": nilai_rapor_semester_4,
            "semester_5": nilai_rapor_semester_5,
            "peringkat_eligible": peringkat_eligible,
            "jumlah_eligible": jumlah_siswa_eligible',
            "jurusan_sekolah": peminatan,
            "jurusan_1": jurusan_1,
            "universitas_1": universitas_1,
            "linearitas_1": linearitas_1,
            "jurusan_2": jurusan_2,
            "universitas_2": universitas_2,
            "linearitas_2": linearitas_2
        }])

        try:

            # Menyesuaikan urutan fitur jika tersedia
            if hasattr(model, "feature_names_in_"):
                input_data = input_data.loc[
                    :, model.feature_names_in_
                ]

            # ==================================
            # PREDIKSI PROBABILITAS
            # ==================================

            probabilities = model.predict_proba(input_data)[0]

            # Mengambil probabilitas kelas diterima (1)
            class_index = list(model.classes_).index(1)

            probability = probabilities[class_index]

            # Mengubah ke persen
            probability_percent = probability * 100

            # ==================================
            # KATEGORI BERDASARKAN 19,6%
            # ==================================

            threshold = 19.6

            if probability_percent > threshold:
                category = "DI ATAS RATA-RATA"
                description = (
                    "Probabilitas penerimaan berada di atas "
                    "rata-rata penerimaan sebesar 19,6%."
                )

            elif probability_percent < threshold:
                category = "DI BAWAH RATA-RATA"
                description = (
                    "Probabilitas penerimaan berada di bawah "
                    "rata-rata penerimaan sebesar 19,6%."
                )

            else:
                category = "SAMA DENGAN RATA-RATA"
                description = (
                    "Probabilitas penerimaan sama dengan "
                    "rata-rata penerimaan sebesar 19,6%."
                )

            # ==================================
            # HASIL PREDIKSI
            # ==================================

            st.divider()

            st.subheader("6. Hasil Prediksi SNBP")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Probabilitas Diterima",
                    value=f"{probability_percent:.2f}%"
                )

            with col2:
                st.metric(
                    label="Rata-rata Penerimaan",
                    value=f"{threshold:.2f}%"
                )

            # ==================================
            # VISUALISASI PERBANDINGAN
            # ==================================

            st.subheader("7. Perbandingan dengan Rata-rata")

            st.write(f"**Kategori: {category}**")

            st.info(description)

            # Progress bar probabilitas
            st.write("Probabilitas penerimaan")

            st.progress(float(probability))

            # ==================================
            # INTERPRETASI
            # ==================================

            st.subheader("8. Interpretasi Hasil")

            if probability_percent > threshold:

                st.success(
                    f"Probabilitas penerimaan sebesar "
                    f"{probability_percent:.2f}%, lebih tinggi "
                    f"dari rata-rata penerimaan sebesar "
                    f"{threshold:.2f}%."
                )

            elif probability_percent < threshold:

                st.warning(
                    f"Probabilitas penerimaan sebesar "
                    f"{probability_percent:.2f}%, lebih rendah "
                    f"dari rata-rata penerimaan sebesar "
                    f"{threshold:.2f}%."
                )

            else:

                st.info(
                    "Probabilitas penerimaan sama dengan "
                    "rata-rata penerimaan."
                )

            # ==================================
            # DETAIL PROBABILITAS SETIAP KELAS
            # ==================================

            with st.expander("Lihat detail probabilitas kelas"):

                for kelas, prob in zip(
                    model.classes_, probabilities
                ):

                    label = (
                        "Diterima" if kelas == 1
                        else "Tidak Diterima" if kelas == 0
                        else str(kelas)
                    )

                    st.write(
                        f"**{label}: {prob * 100:.2f}%**"
                    )

        except Exception as e:

            st.error(
                f"Terjadi kesalahan saat prediksi: {e}"
            )
