import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Load model
model_path = 'model.h5'  # Adjust path based on your project structure
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)

# Function to prepare image
def prepare_image(image, target_size=(128, 128)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Adding batch dimension
    return image_array

# Streamlit app code
st.title("Image Classification")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    prepared_image = prepare_image(image)
    preds = model.predict(prepared_image)
    pred_class = np.argmax(preds, axis=1)
    
    st.write(f"Predicted Class: {pred_class[0]}")
