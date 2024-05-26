import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
from PIL import Image
import tempfile

# Load the saved model
saved_model_path = '/Users/mananhingorani/4. DeepFake_Detection/output/final_model2.keras'
model = load_model(saved_model_path)

# Function to process and predict the image
def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    if predictions[0, 0] > 0.5:
        return "Real Image"
    else:
        return "Fake Image"

# Streamlit app
st.title("Deepfake Detection Web App")

st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

st.header("Upload an Image to Check if it's a Deepfake")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Prediction
    label = predict_image(img)
    st.write(f"Prediction: **{label}**")
