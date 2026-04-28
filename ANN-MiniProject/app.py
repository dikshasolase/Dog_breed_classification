import streamlit as st
import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

# Page config
st.set_page_config(
    page_title="Dog Breed Detector",
    layout="centered"
)

# Hide sidebar
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Load PyTorch model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.eval()
    return model

model = load_model()

# Image preprocessing (PyTorch style)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# UI
st.title("🐶 AI Dog Breed Detection System")
st.write("Upload a dog image and the model will try to identify the breed.")

uploaded_file = st.file_uploader(
    "Choose a dog image...",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)
    img = img.unsqueeze(0)

    # Prediction
    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        top5_prob, top5_catid = torch.topk(probs, 3)

    # Labels (ImageNet classes)
    import json
    import urllib.request

    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = urllib.request.urlopen(LABELS_URL).read().decode("utf-8").split("\n")

    results = []
    for i in range(3):
        results.append((labels[top5_catid[i]], top5_prob[i].item()))

    # Filter dog-related results
    dog_keywords = ["dog", "retriever", "shepherd", "terrier", "hound", "spaniel"]
    filtered = [r for r in results if any(k in r[0].lower() for k in dog_keywords)]

    if filtered:
        top_label = filtered[0][0]
        top_prob = filtered[0][1] * 100
    else:
        top_label = results[0][0]
        top_prob = results[0][1] * 100

    st.success(f"Predicted Breed: {top_label}")
    st.write(f"Confidence: {top_prob:.2f}%")

else:
    st.info("Please upload an image to get started.")
