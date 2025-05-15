import streamlit as st
import pandas as pd 

st.title("TF-IDF")
st.write("Term Frequency – Inverse Document Frequency (TF-IDF) atau pembobotan data dilakukan setelah proses preprocessing data, dimana kata atau teks dasar diproses untuk dijadikan sebuah vektor. TF-IDF adalah teknik pengolahan teks yang digunakan untuk memberikan bobot pada kata-kata dalam dokumen. Tujuan TF-IDF adalah untuk mengidentifikasi kata-kata yang paling penting dalam dokumen atau kumpulan dokumen. Nilai frekuensi kata yang muncul dalam dokumen disebut Frequency Term (TF).")
st.write("Pada tabel dibawah adalah fitur yang sudah dilakukan pembobotan")
pre = pd.read_csv("data/50_tfdf.csv")
pre = pre.head(10)

st.dataframe(pre)