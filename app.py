import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import os

# Load model
model_path = 'model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)

# Function to prepare image prediction
def prepare_image_and_predict(image, model):
    image = load_img(image_data, target_size=(150, 150))
    if image.mode != "RGB":
       image = image.convert("RGB")
    image = np.asarray(image)
    img_array = np.expand_dims(image, axis=0)
    img_array /= 255.0
    img_reshape = img_array[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    
    return prediction

# Streamlit app
st.title("Image Classification with MobileNet")
st.write("Upload an image for classification")

uploaded_file = st.file_uploader("Choose and image...", type=["jpeg", "jpg", "png"])


if uploaded_file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    prediction = prepare_image_and_predict(image, model)
    
    try:
        class_names = ['Cheetah', 'Lion']
        pred_class = np.argmax(prediction, axis=1)[0]

        st.write(f"Image is a {class_names[pred_class]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
