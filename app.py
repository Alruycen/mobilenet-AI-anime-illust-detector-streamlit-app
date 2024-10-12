
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
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
    def rescale(image):
        return (image / 127.5) - 1.0
    
    if submit:
        with st.spinner('Please wait for a moment, the model is working...'):
            if best_model is None:
                best_model = keras.models.load_model('ai_detection_best_model.h5')

            resized_data = data.resize((224, 224))
            
            # ganti ke array dan convert ke RGB
            rgb_data = np.array(resized_data.convert("RGB"))

            # augmentation
            test_gen = ImageDataGenerator(
                preprocessing_function=rescale,
                rotation_range=20,
                zoom_range=0.2,
                shear_range=0.2,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True
            )

            # dapetin data test dari augmentation
            augmented_iter = test_gen.flow(np.expand_dims(rgb_data, axis=0), batch_size=1)
            
            augmented_images = [augmented_iter.next()[0] for _ in range(9)]
            
            # test-time augmentation
            def tta_predict(model, data_list):
                yhat = [model.predict(np.expand_dims(img, axis=0)) for img in data_list]
                return np.mean(yhat, axis=0)
            
            # Hasil Prediksi
            y_prob = tta_predict(best_model, augmented_images)
            
        st.success('Finish!')
        
        # seperti yang disebutkan di Data Preprocessing, AI dilabeli 0, 
        # sementara non-AI dilabeli 1 secara otomatis
        
        # dalam kasus ini, 100 - (probability non-AI, karena non-AI mendekati 1, 
        # sementara AI mendekati 0)
        hasil = 100 - (np.round(y_prob[0], 2) * 100)
        string_hasil = ('AI-generated' if hasil > 50 else 'not AI-generated')
        
        # kalau lebih dari sama dengan 50 persen kemungkinannya tinggi buatan AI, 
        # sementara di bawah 50 persen berarti belom tentu
        
        st.write('Based on model prediction, this illustration is ', string_hasil)
    
        st.write('AI-generated features detected in percentages: ')
        st.write(hasil)
