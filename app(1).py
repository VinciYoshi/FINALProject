#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

model = load_model('model.h5')

def prepare_image(image, target_size=(128, 128)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

st.set_page_config(
    page_title="Image Classification: Lion or Cheetah",
    page_icon=":camera:",
    layout="centered",
    initial_sidebar_state="auto") 

background_color = "#8B4513"  # Brown
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {background_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Image Classification: Lion or Cheetah")
st.write("Upload an image for classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
  
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    st.write("Classifying...")
    
    prepared_image = prepare_image(image)
    
    preds = model.predict(prepared_image)
    pred_class = np.argmax(preds, axis=1)
    
    st.write(f"Predicted Class: {pred_class[0]}")

    st.write("Prediction Confidence Scores:")
    for idx, score in enumerate(preds[0]):
        st.write(f"Class {idx}: {score:.4f}")
else:
    st.write("Please upload an image file to proceed.")
