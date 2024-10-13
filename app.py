
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

st.set_page_config(page_title='Detect AI-generated Anime Illustrations',
                   layout='wide')

st.header('Detect Anime Illustrations produced by Generative AI')

col1, col2, col3 = st.columns(3)

best_model = None
n_aug = 10

# function untuk upload gambar dalam try-catch
def unggah_gambar():
    if submit:
        try:            
            img = Image.open(file)
    
            return img
        except:
            st.warning("There is some error happening when uploading the image.")

def load_best_model():
    global best_model
    if best_model is None:
        with st.spinner("Loading model..."):
            best_model = keras.models.load_model('ai_detection_best_model', custom_objects=None, compile=True)
    return best_model

def rescale(image):
    return (image / 127.5) - 1.0
            
# augmentation
test_gen = ImageDataGenerator(
    preprocessing_function=rescale,
    rotation_range=20,  
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],  
    channel_shift_range=20.0,
    shear_range=0.2, 
    width_shift_range=0.1, 
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
)

def rescale_for_plotting(image):
    return (image + 1) / 2   

def plot_images(rescaled_images, n):
    with st.spinner("Augmenting images..."):
        fig, ax = plt.subplots(2, 5, figsize=(8, 8))
        for i in range(n_aug):
            ax[i // 5, i % 5].imshow(rescaled_images[i])
            ax[i // 5, i % 5].axis("off")
        plt.tight_layout()
        st.pyplot(fig) 

# test-time augmentation
def tta_predict(model, images):
    yhat = [model.predict(np.expand_dims(img, axis=0)) for img in images]
    return tf.reduce_mean(yhat, axis=0)

with col1:
    form = st.form("form_data")

    file = form.file_uploader("Upload the image here...", 
                              type=['png','jpeg','jpg'])

    submit = form.form_submit_button("Submit result")

    data = unggah_gambar()
    
with col2:
    if submit:
        with st.spinner():
            st.image(data)
        
with col3:
    if submit:
        best_model = load_best_model()
        
        with st.spinner('Please wait for a moment, the model is working...'):
            # Resize image
            resized_data = data.resize((224,224))

            # Change to arr and convert to 3 channel (RGB)
            rgb_data = np.array(resized_data.convert("RGB"))

            # dapetin data test dari augmentation
            augmented_iter = test_gen.flow(np.expand_dims(rgb_data, axis=0), batch_size=1)
            
            augmented_images = [augmented_iter.next()[0] for _ in range(n_aug - 1)]

            augmented_images.append(rescale(rgb_data))

            # dapetin gambar dari data test augmented
            rescaled_images = [rescale_for_plotting(img) for img in augmented_images]

            # Plot images
            plot_images(rescaled_images, n_aug)
            
            # Hasil Prediksi
            y_prob = tta_predict(best_model, augmented_images)
            
        st.success('Finish!')
        
        # seperti yang disebutkan di Data Preprocessing, AI dilabeli 0, 
        # sementara non-AI dilabeli 1 secara otomatis
        
        # dalam kasus ini, 100 - (probability non-AI, karena non-AI mendekati 1, 
        # sementara AI mendekati 0)
        hasil = 100 - (np.round(y_prob[0], 2) * 100)
        string_hasil = ('AI-generated' if tf.math.round(y_prob) == 0 else 'not AI-generated')
        
        # kalau di atas 0.4 berkemungkinan merupakan buatan AI, 
        # sementara di bawah 0.4 berarti belom tentu
        
        st.write('Based on model prediction, this illustration is ', string_hasil)
    
        st.write('AI-generated features detected in percentages: ')
        st.write(hasil)
