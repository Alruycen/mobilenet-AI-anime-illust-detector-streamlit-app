
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras

st.set_page_config(page_title='Detect AI-generated Anime Illustrations',
                   layout='wide')

st.header('Detect Anime Illustrations produced by Generative AI')

col1, col2, col3 = st.columns(3)

best_model = None # buat declare model

with col1:
    form = st.form("form_data")

    file = form.file_uploader("Upload the image here...", 
                                      type=['png','jpeg','jpg'])
    
    # function untuk upload gambar dalam try-catch
    def unggah_gambar():
        if submit:
            try:            
                img = Image.open(file)
        
                return img
            except:
                st.warning("There is some error happening when uploading the image.")

    submit = form.form_submit_button("Submit result")

    data = unggah_gambar()
    
with col2:
    if submit:
        with st.spinner():
            st.image(data)
        
with col3:
    if submit:
        with st.spinner('Please wait for a moment, the model is working...'):
            if best_model is None:
                best_model = keras.models.load_model('ai_detection_best_model.h5')
            
            # Resize image
            resized_data = data.resize((256, 256))

            # Change to arr and convert to 3 channel (RGB)
            rgb_data = np.array(resized_data.convert("RGB"))
        
            # Rescale
            rescaled_data = img_to_array(rgb_data) * 1./255
        
            # Numpy Array
            data_arr = np.array([rescaled_data])  
            
            # Hasil prediksi            
            y_prob = best_model.predict(data_arr) 
            
        st.success('Finish!')
        
        # seperti yang disebutkan di Data Preprocessing, AI dilabeli 0, 
        # sementara non-AI dilabeli 1 secara otomatis
        
        # dalam kasus ini, 100 - (probability non-AI, karena non-AI mendekati 1, 
        # sementara AI mendekati 0)
        hasil = 100 - (np.round(y_prob[0], 2) * 100)
        string_hasil = ("AI-generated" if hasil > 50 else "not AI-generated")
        
        # kalau lebih dari sama dengan 50 persen kemungkinannya tinggi buatan AI, 
        # sementara di bawah 50 persen berarti belom tentu
        
        st.write('Based on model prediction, this illustration is ', string_hasil)
    
        st.write('AI-generated features detected: ')
        st.write(hasil)
