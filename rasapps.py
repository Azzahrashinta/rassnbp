
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
            "semester_1": semester_1,
            "semester_2": semester_2,
            "semester_3": semester_3,
            "semester_4": semester_4,
            "semester_5": semester_5,
            "peringkat_eligible": peringkat_eligible,
            "jumlah_eligible": jumlah_eligible,
            "jurusan_sekolah": jurusan_sekolah,
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
