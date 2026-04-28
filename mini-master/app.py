import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import urllib.request

st.set_page_config(page_title="Dog Breed Detector", layout="centered")

# Hide sidebar
st.markdown("<style>section[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Using MobileNetV2 for a good balance of speed and accuracy
    model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
    model.eval()
    return model

@st.cache_resource
def load_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
    return labels

model = load_model()
labels = load_labels()

# Standard ImageNet transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🐶 AI Dog Breed Detection System")
st.write("Upload a photo of a dog to identify its breed.")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process Image
    img_t = transform(image).unsqueeze(0)

    with st.spinner("Analyzing breed..."):
        with torch.no_grad():
            output = model(img_t)
            probs = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 to ensure we catch a dog label if it's there
    top_probs, top_idxs = torch.topk(probs, 5)
    results = [(labels[i], top_probs[j].item()) for j, i in enumerate(top_idxs)]

    # Filter for dogs (ImageNet has many specific breed names without the word 'dog')
    # This checks if the predicted class index falls within the known dog ranges in ImageNet (roughly 151-268)
    dog_results = [(name, p) for name, p in results if 151 <= labels.index(name) <= 268]

    if dog_results:
        label, prob = dog_results[0]
        st.success(f"**Predicted Breed:** {label.replace('_',' ').title()}")
        st.metric("Confidence", f"{prob*100:.2f}%")
    else:
        # Fallback if no dog is detected in top 5
        label, prob = results[0]
        st.warning("This might not be a dog, but here is the closest match:")
        st.write(f"**Object:** {label.title()} ({prob*100:.2f}%)")

else:
    st.info("Please upload an image to begin.")
