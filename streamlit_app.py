import streamlit as st

st.set_page_config(
    page_title="Analisis Clustering Cacat Produk",
    layout="wide"
)

st.title("Analisis Clustering Cacat Produk pada Industri Manufaktur")

st.write(
    "Aplikasi ini digunakan untuk menampilkan hasil analisis clustering "
    "terhadap data cacat produk pada industri manufaktur."
)

st.divider()

st.subheader("Identitas Mahasiswa")

st.write("Nama : ytta")
st.write("NIM : belum tau")
st.write("Mata Kuliah : gatau")

st.divider()

st.subheader("Deskripsi Aplikasi")

st.write(
    "Analisis clustering digunakan untuk mengelompokkan data cacat produk "
    "berdasarkan kemiripan karakteristik yang dimiliki. Hasil pengelompokan "
    "diharapkan dapat membantu dalam mengidentifikasi pola cacat produk dan "
    "menentukan tindakan perbaikan yang sesuai."
)

st.info(
    "Dataset dan hasil analisis clustering akan ditambahkan setelah seluruh "
    "bahan analisis tersedia."
)