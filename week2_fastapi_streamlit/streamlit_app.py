
import requests
import streamlit as st
from PIL import Image

st.title("MNIST Digit Predictor")
st.write("Upload an mnist digit image to get the prediction from the FastAPI model.")

backend_url = "http://127.0.0.1:8000/predict_image"

uploaded_file=st.file_uploader("choose an image",type=["png","jpg","jpeg"])

if uploaded_file:
    image=Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",use_column_width=True)
    files={"file":uploaded_file.getvalue()}
    response=requests.post(backend_url,files=files)
    if response.status_code==200:
        result=response.json()
        st.success(f"Prediction: {result['prediction']} with confidence {result['confidence']}")
    else:
        st.error("Error in prediction. Please try again.")