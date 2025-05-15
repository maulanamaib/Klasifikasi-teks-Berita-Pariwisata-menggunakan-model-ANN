import streamlit as st
import model
import re
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()



st.title("Klasifikasi Teks Berita pariwisata dengan ANN")
st.subheader("Masukkan teks untuk diklasifikasikan")
def preprocess(text):
    # Remove special characters and numbers
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'@[A-Za-z0-9]+',' ',text)
    text = re.sub(r'_x000D__x000D__x000D_',' ',text)
    text = re.sub(r'SCROLL TO CONTINUE WITH CONTENT_x000D_',' ',text)
    text = re.sub(r'#[A-Za-z0-9]+',' ',text)
    text = re.sub(r"http\S+",' ',text)
    text = re.sub(r'[0-9]+',' ',text)
    text = re.sub(r'\n',' ',text)
    text = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", text)
    text = text.strip(' ')
    text = text.lower()
    # Menghapus stopwords
    text = stopword_remover.remove(text)
    # Stemming
    text = word_tokenize(text)
    return text
# Input dari pengguna
input_text = st.text_area("Teks", "Masukkan teks di sini")

if st.button("Klasifikasikan"):
    if input_text.strip():
        # Preprocessing
        st.write("Melakukan preprocessing...")
        # preprocessed_text = model.preprocess(input_text)
        st.write("Teks setelah preprocessing:")
        # st.write(preprocessed_text)


        # Normalisasi
        st.write("Melakukan pembobotan...")
        normalized_data = model.normalisasi(input_text)
        # st.write(normalized_data)
        # Prediksi
        st.write("Melakukan prediksi...")
        prediction = model.ann(normalized_data)
        cat = [ "alam","buatan","budaya","bukan pariwisata"]
        # Menampilkan hasil prediksi
        st.write("Hasil prediksi:")
        st.markdown(f"<h3 style='color:green;'>{cat[prediction]}</h3>", unsafe_allow_html=True)
        
    else:
        st.error("Harap masukkan teks untuk diklasifikasikan.")