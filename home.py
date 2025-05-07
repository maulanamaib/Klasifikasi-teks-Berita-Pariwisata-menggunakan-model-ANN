import streamlit as st
import pandas as pd


st.title("Klasifikasi Teks Berita Pariwisata")
st.write("Pariwisata adalah suatu lokasi wisata yang menarik untuk wisatawan berkunjung. Pariwisata sendiri memiliki beberapa kategori sendiri seperti wisata alam, wisata budaya dan wisata buatan yang didasarkan pada usaha daya tarik wisata. Berdasarkan pada undang-undang republik Indonesia nomor 10 tahun 2009 tentang kepariwisataan pasal 14 ayat 1 yang berbunyi “Yang dimaksud dengan “usaha daya tarik wisata” adalah usaha yang kegiatannya mengelola daya tarik wisata alam, daya tarik wisata budaya, dan daya tarik wisata buatan/binaan manusia”. Pariwisata beberapa diangkat dari website berita, salah satu website yang mengangkat berita pariwisata adalah detik dimana berita yang di angkat sangat banyak hingga ribuan. Data yang saya gunakan berupa data berita pariwisata yang di dapat dari website tersebut. ")
st.write("Klasifikasi merupakan salah satu teknik dalam mechine learning yang digunakan untuk melakukan pemisahan data ke dalam beberapa kelas. Proses klasifikasi adalah proses memperoleh suatu fungsi atau pola yang menjelaskan atau memisahkan konsep/kelas informasi")
st.write("Dalam melakukan klasifikasi teks menggunakan pendekatan Artificial Neural Network (ANN) untuk menghitung bobot yang didapatkan dari TF-IDF. Artificial Neural Network (ANN) sendiri adalah sebuah model komputasi terinspirasi dari sistem saraf biologis yang dapat belajar dari data dan menghasilkan keluaran yang diinginkan dengan menyesuaikan bobot koneksi antar neuron[22]. Dalam metode Artificial Neural Network (ANN) ada banyak istilah yakni bobot (weight), lapisan (layer), neuron, masukan (input), keluaran (output), dan lapisan tersembunyi (hidden layers).")
df = pd.read_excel("data/data_skripsi_5k.xlsx")
df = df.head(10)
st.dataframe(df)