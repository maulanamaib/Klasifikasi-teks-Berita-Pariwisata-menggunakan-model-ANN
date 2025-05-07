import streamlit as st
import pandas as pd 

st.title("Preprocessing")
st.write("Tahap preprocessing adalah langkah setelah mendapatkan data, tahap ini tidak kalah penting untuk dilkakukan karena setelah melakukan preprocessing akan masuk ke dalam tahap model. Preprocessing adalah salah satu tahap pada mechine learning dimana data akan di bersihkan agar mengurangi terjadinya noise pada data yang menyebabkan akurasi menjadi kecil. Sehingga dengan proses preprosessing data ini dapat meningkatkan akurasi serta performa dari metode karena metode pada machine learning belajar dari data-data yang relevan.")

pre = pd.read_csv("data/prepro.csv")
pre = pre.head(10)

st.dataframe(pre)