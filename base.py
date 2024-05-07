import streamlit as st
from PIL import Image
import numpy as np
import pickle

# Define a function to load the model
@st.cache()
def load_model():
    with open('models/random_forest_model.pkl', 'rb') as model_file:
        loaded_rf_model = pickle.load(model_file)
    return loaded_rf_model

# Load the trained model from disk
loaded_rf_model = load_model()

def preprocess_image(image):
    processed_image = np.array(image.convert('L').resize((224, 224))).flatten().reshape(1, -1)
    return processed_image

def predict(image):
    processed_image = preprocess_image(image)
    prediction = loaded_rf_model.predict(processed_image)
    return prediction


st.title("Chest CT Scan Classification")
st.write("Upload an image for classification.")

uploaded_file = st.file_uploader("Choose a chest CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')

    if st.button('Predict'):
        prediction = predict(image)
        st.write("Prediction:", prediction)
