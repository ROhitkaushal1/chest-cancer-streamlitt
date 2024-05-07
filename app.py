import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from pycaret.classification import load_model

# Define a function to load the model
@st.cache(allow_output_mutation=True)
def load_pycaret_model():
    loaded_model = load_model('models/pycaret')
    return loaded_model

# Load the PyCaret model
loaded_model = load_pycaret_model()

def preprocess_image(image):
    # Resize the image to 224x224 and convert to grayscale
    resized_image = image.resize((224, 224)).convert('L')
    # Flatten the image array and reshape to (1, -1) for prediction
    processed_image = np.array(resized_image).flatten().reshape(1, -1)
    return processed_image

def predict(image):
    processed_image = preprocess_image(image)
    # Create a DataFrame with the processed image array
    df = pd.DataFrame(processed_image)
    # Make prediction using the loaded PyCaret model
    prediction = loaded_model.predict(df)
    return prediction[0]

st.title("Chest CT Scan Classification")
st.write("Upload an image for classification.")

uploaded_file = st.file_uploader("Choose a chest CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')

    if st.button('Predict'):
        prediction = predict(image)
        st.write("Prediction:", prediction)
