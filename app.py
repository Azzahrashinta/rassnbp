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
