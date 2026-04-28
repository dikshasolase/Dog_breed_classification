import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# Page config
st.set_page_config(
    page_title="Dog Breed Detector",
    layout="centered"
)

# Hide sidebar completely (optional but included)
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Load pretrained model
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# UI
st.title("🐶 AI Dog Breed Detection System")
st.write("Upload a dog image and the model will try to identify the breed.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a dog image...",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img)

    # Handle RGBA images
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Prediction
    with st.spinner("Analyzing image..."):
        preds = model.predict(img)
        results = decode_predictions(preds, top=3)[0]

    # Filter dog-related predictions
    dog_keywords = ["dog", "retriever", "shepherd", "terrier", "hound", "spaniel"]
    filtered = [r for r in results if any(k in r[1] for k in dog_keywords)]

    # Select best result
    if filtered:
        top_label = filtered[0][1]
        top_prob = filtered[0][2] * 100
    else:
        top_label = results[0][1]
        top_prob = results[0][2] * 100

    # Display result
    st.success(f"Predicted Breed: {top_label.replace('_',' ').title()}")
    st.write(f"Confidence: {top_prob:.2f}%")

else:
    st.info("Please upload an image to get started.")
