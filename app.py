import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import os

# Load model
model_path = 'model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)

# Function to prepare image for prediction
def prepare_image(image, target_size=(150, 150)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Streamlit app code
st.title("Image Classification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    prepared_image = prepare_image(image)
    prediction = model.predict(prepared_image)
    class_names = ['Cheetah', 'Lion']  # Adjust according to your model's classes
    pred_class = np.argmax(prediction, axis=1)[0]

    st.write(f"Image is a {class_names[pred_class]}")
