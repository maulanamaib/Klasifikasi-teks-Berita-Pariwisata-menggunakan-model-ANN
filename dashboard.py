import streamlit as st

# from SessionState import _get_state


# state = _get_state()

home = st.Page("home.py", title="Dashboard", icon=":material/home:")
prepro = st.Page("prepro.py", title="Preprocesing", icon=":material/rebase:")
tfdf = st.Page("tfdf.py", title="TF-IDF", icon=":material/repeat_one:")
klasifikasi = st.Page("klasifikasi.py", title="klasifikasi", icon=":material/monitoring:")
app = st.Page("app.py", title="Uji Coba", icon=":material/brand_family:")

pg = st.navigation([home, prepro, tfdf, klasifikasi, app])
st.set_page_config(page_title="Klasifikasi Teks", page_icon=":material/edit:")
pg.run()