#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/weights-improvement-71-0.88.hdf5")
    return model

model = load_model()

st.write("""
# Image Class Predictor
""")

file = st.file_uploader("Choose an image file", type=['jpg', 'png', 'jpeg'])

def import_and_predict(image_data, model):
    size = (64, 64) 
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = img / 255.0  
    img_reshape = np.expand_dims(img, axis=0)  
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Cheetah', 'Lion'] 
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Prediction Confidence: {np.max(prediction) * 100:.2f}%")
