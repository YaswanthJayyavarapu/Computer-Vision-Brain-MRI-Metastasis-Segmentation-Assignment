# streamlit_app.py

import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose a brain MRI image...", type="png")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    
    # Call the FAST API backend for prediction
    files = {'file': uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    prediction = np.array(response.json()["prediction"])
    
    # Display the prediction
    st.image(prediction, caption="Predicted Metastasis", use_column_width=True)
