
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras

st.set_page_config(page_title='Deteksi Ilustrasi Anime Hasil Produksi AI generatif',
                   layout='wide')

st.header('Deteksi Ilustrasi Anime Hasil Produksi AI generatif')

col1, col2, col3 = st.columns(3)

best_model = None # buat declare model

with col1:
    form = st.form("form_data")

    file = form.file_uploader("Unggah gambar di sini...", 
                                      type=['png','jpeg','jpg'])
    
    # function untuk upload gambar dalam try-catch
    def unggah_gambar():
        if submit:
            try:            
                img = Image.open(file)
        
                return img
            except:
                st.warning("Terjadi kesalahan saat mengunggah gambar.")

    submit = form.form_submit_button("Cek hasil")

    data = unggah_gambar()
    
with col2:
    if submit:
        st.image(data)
        
with col3:
    if submit:
        with st.spinner('Mohon ditunggu, model sedang bekerja...'):
            if best_model is None:
                best_model = keras.models.load_model('ai_detection_best_model.h5')
            
            # Resize image
            resized_data = data.resize((224, 224))

            # Change to arr and convert to 3 channel (RGB)
            rgb_data = np.array(resized_data.convert("RGB"))
        
            # Rescale
            rescaled_data = img_to_array(rgb_data) * 1./255
        
            # Numpy Array
            data_arr = np.array([rescaled_data])  
            
            # Hasil prediksi            
            y_prob = best_model.predict(data_arr) 
            
        st.success('Selesai!')
        
        # seperti yang disebutkan di Data Preprocessing, AI dilabeli 0, 
        # sementara non-AI dilabeli 1 secara otomatis
        
        # dalam kasus ini, 100 - (probability non-AI, karena non-AI mendekati 1, 
        # sementara AI mendekati 0)
        hasil = 100 - (np.round(y_prob[0], 2) * 100)
        string_hasil = ("buatan AI" if hasil > 50 else "bukan buatan AI")
        
        # kalau lebih dari sama dengan 50 persen kemungkinannya tinggi buatan AI, 
        # sementara di bawah 50 persen berarti belom tentu
        
        st.write('Menurut model, gambar di samping ', string_hasil)
    
        st.write('Berikut merupakan persentasenya: ')
        st.write(hasil)
