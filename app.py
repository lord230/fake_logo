import streamlit as st
st.set_page_config(page_title="Fake Logo Detector", page_icon="🧠", layout="centered")
import torch

import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224
threshold = 0.7

# --- Class Names ---
class_names = ["real", "fake"]

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    weights_path = "results/efficientnet_fake_logo.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# --- Prediction ---
def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, 1)
        return pred_class.item(), confidence.item()



st.markdown(
    """
    <style>
    .main {
        background-color: #f2f6fa;
    }
    .title {
        font-size: 38px;
        text-align: center;
        font-weight: bold;
        background: -webkit-linear-gradient(90deg, #3a7bd5, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
    }
    .footer {
        text-align: center;
        font-size: 13px;
        color: gray;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="title">🧠 Fake Logo Detector</div>', unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses a fine-tuned **EfficientNet-B0** model to detect whether a logo is **real** or **AI-generated (fake)**.
    
    - Upload any logo image
    - Hit **Predict**
    - Get prediction and confidence
    
    Created by `LORD` 🧑‍💻
    """)

# Upload image
uploaded_file = st.file_uploader("Upload a logo image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Logo", use_container_width=True)

    image_tensor = preprocess_image(image)

    if st.button("🔍 Predict"):
        pred_idx, confidence = predict(image_tensor)
        label = class_names[pred_idx]
        st.markdown("---")
        st.subheader("🧾 Prediction Result")
        st.markdown(f"**Label:** `{label.upper()}`")

        if confidence >= threshold:
            st.success(f"✅ Percentage : {confidence:.2%}")
        else:
            st.warning(f"⚠️ Low Percentage: {confidence:.2%} (below {threshold:.0%} threshold)")

# Footer
st.markdown('<div class="footer">© 2025 Fake Logo Detector by LORD</div>', unsafe_allow_html=True)
