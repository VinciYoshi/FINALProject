import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

def import_and_predict(image_data, model):
    size = (224, 224)  # Example size, adjust to your model's requirement
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0  # Normalize the image
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Load your model
model = tf.keras.models.load_model('path/to/your/model')

# Streamlit code to handle file upload and display
file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Dog', 'Other']  # Replace with your actual class names
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
