import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_model.h5"

model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/indices.json"))


def load_preprocess(img_path, target_dim=(224, 224)):
    img = Image.open(img_path)
    img = img.resize(target_dim)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255
    return img


def predict_class(models, img_path, class_ind):
    df = load_preprocess(img_path)
    predicted = models.predict(df)
    index = np.argmax(predicted, axis=1)[0]
    class_name = class_ind[str(index)]
    return class_name


st.title("Plant Disease Classification")

upload_image = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if upload_image is not None:
    image = Image.open(upload_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_image = image.resize((255, 255))
        st.image(resized_image)

    with col2:
        if st.button('Classify'):
            prediction = predict_class(model, upload_image, class_indices)
            st.success(f"Prediction: {str(prediction)}")
