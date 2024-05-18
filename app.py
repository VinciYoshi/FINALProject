#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

# Load model
model_path = 'model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)

# Function to prepare image
def prepare_image(image, target_size=(128, 128)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Streamlit app
st.title("Image Classification with MobileNet")
st.write("Upload an image for classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Classifying...")
    prepared_image = prepare_image(image)
    
    try:
        preds = model.predict(prepared_image)
        pred_class = np.argmax(preds, axis=1)
        st.write(f"Predicted Class: {pred_class[0]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
