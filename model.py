import joblib
import pandas as pd
import numpy as np
import re
import os
import streamlit as st
import tensorflow as tf
import nltk
# from tensorflow import keras
# import keras
# from tensorflow.keras.models import load_model
# from keras.models import load_model
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# model = load_model('model/model_ann.h5')
# st.write(model.summary())
# Fungsi preprocessing
def preprocess(text):
    # Validasi input harus string
    if not isinstance(text, str):
        raise ValueError("Input preprocess harus berupa string.")
    
    # Hapus karakter dan pola tertentu
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Hapus karakter non-ASCII
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text) # Hapus mention
    text = re.sub(r'#\w+', ' ', text)           # Hapus hashtag
    text = re.sub(r"http\S+", ' ', text)        # Hapus URL
    text = re.sub(r'[0-9]+', ' ', text)         # Hapus angka
    text = re.sub(r'\n', ' ', text)             # Hapus newline
    text = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", text)  # Hapus simbol
    text = text.strip()  # Hapus spasi di awal/akhir
    text = text.lower()  # Ubah ke huruf kecil
    
    # Hapus stopword
    text = stopword_remover.remove(text)
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Gabungkan token kembali ke string jika diperlukan
    # text = ' '.join(tokens)
    
    return tokens

def normalisasi(x):
    # print(x)
    # Pastikan x adalah list, lalu buat DataFrame dengan kolom 'Konten'
    if isinstance(x, str):  # Jika x adalah teks tunggal
        x = [x]  # Ubah menjadi list
    elif not isinstance(x, list):  # Jika x bukan list atau teks
        raise ValueError("Input harus berupa teks atau list teks.")
    
    cols = ['Konten']
    df = pd.DataFrame(x, columns=cols)  # Membuat DataFrame dari input
    # print(df)
    # Import data_test
    data_test = pd.read_csv('data/data_test.csv')
    
    # Terapkan preprocess ke kolom 'Konten'
    
    
    # Gabungkan data_test dengan df
    data_test = pd.concat([data_test["Konten"], df])
    data_test = data_test['Konten'].apply(preprocess)
    # print(data_test)
    data_train = open('model/X_train_columns.txt', 'r')
    data_train = data_train.read()
    data_train = data_train.split("\n")
    # data_train = print(len(data_train))
    # print((df))
    # return data_train
    # Transformasikan data dengan tf-idf
    # print(data_test)
    tfidf = joblib.load('model/tfidf.sav')
    tfdf = tfidf.fit_transform([' '.join(x) for x in data_test]).toarray()
    feature_names = tfidf.get_feature_names_out()
    tf_df = pd.DataFrame(tfdf, columns=feature_names)

    # Buat daftar fitur gabungan (union dari feature_names dan data_train)
    all_features = list(set(feature_names).union(data_train))

    # Inisialisasi DataFrame baru dengan nilai 0.0
    merged_df = pd.DataFrame(0.0, index=tf_df.index, columns=all_features)

    # Isi kolom yang ada di feature_names dengan nilai dari tf_df
    merged_df[feature_names] = tf_df

    # Sesuaikan urutan kolom dengan data_train
    merged_df = merged_df[data_train]

    # print(merged_df)
    return merged_df


def ann(x):
    model_path = "model/model_ann.h5"  # Gantilah dengan path absolut yang benar
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        y_pred = model.predict(x)
        y_pred_classes = np.argmax(y_pred, axis=1)

        return y_pred_classes[-1]
    else:
        print(f"File {model_path} tidak ditemukan.")
        return None
