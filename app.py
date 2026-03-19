import streamlit as st
import cv2
import numpy as np
import torch
import os
import tempfile
from feature_extractor import extract_all_features, extract_noise_map
from quantum_model import HybridConcatModel
import pennylane as qml

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="Q-Forensics", layout="wide")
st.title("🛡️ Q-Forensics: Hybrid Quantum-Classical Deepfake Detection")

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    weights_path = "weights/hybrid_concat_weights.pth"
    if not os.path.exists(weights_path):
        st.sidebar.error(f"❌ Trained weights not found at {weights_path}. Please train the model first.")
        return None
    # Determine input dimension from a dummy call (but we'll set it directly)
    # The input dimension is 576 (MobileNet) + 2 (freq) = 578
    model = HybridConcatModel(input_dim=578, reduced_dim=4, n_layers=6, hidden=64)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    st.sidebar.success("✅ Loaded trained model weights")
    return model

model = load_model()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.info("Adjust sensitivity. Lower = more sensitive (more images flagged as fake).")

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader("Upload a face image for analysis...", type=["jpg", "png", "jpeg"])

# -------------------------------
# Helper: generate heatmap
# -------------------------------
def generate_heatmap(original_img, noise_map):
    """Overlay noise residuals onto original image as a heatmap."""
    # Normalize noise map
    noise_norm = cv2.normalize(np.abs(noise_map), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(noise_norm, cv2.COLORMAP_JET)

    # Ensure original is RGB and resized
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    original_resized = cv2.resize(original_img, (256, 256))

    overlay = cv2.addWeighted(original_resized, 0.6, heatmap_color, 0.4, 0)
    return overlay

# -------------------------------
# Main processing
# -------------------------------
if uploaded_file is not None and model is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Load image for display
        file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Extract features
        with st.spinner("Extracting forensic features..."):
            features, noise_map = extract_all_features(tmp_path)   # features: (578,) tensor
            features = features.unsqueeze(0)                        # add batch dimension

        # Run model
        with st.spinner("Quantum circuit processing..."):
            with torch.no_grad():
                logit = model(features).item()
                prob = torch.sigmoid(torch.tensor(logit)).item()
                confidence = prob * 100
                verdict = "FAKE" if prob > threshold else "REAL"

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image_rgb, caption="Original Image", use_container_width=True)

        with col2:
            # Frequency spectrum for display
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (256, 256))
            f = np.fft.fft2(gray_resized)
            fshift = np.fft.fftshift(f)
            mag = 20 * np.log(np.abs(fshift) + 1)
            st.image(mag, caption="Frequency Fingerprint (FFT)", use_container_width=True, clamp=True)

        with col3:
            # Spatial noise residuals
            noise_display = cv2.normalize(np.abs(noise_map), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(noise_display, caption="Spatial Noise Residuals", use_container_width=True, clamp=True)

        # Forensic heatmap
        with st.expander("🔍 Deep Dive: Forensic Heatmap"):
            heatmap = generate_heatmap(image_rgb, noise_map)
            st.image(heatmap, caption="Manipulation Heatmap (Red = high artifacts)", use_container_width=True)
            st.info("The heatmap highlights regions with abnormal noise patterns, often indicative of generative AI upsampling.")

        # Final verdict
        st.divider()
        if verdict == "FAKE":
            st.error(f"🚨 **FAKE DETECTED** | Confidence: {confidence:.2f}%")
            st.warning("This image exhibits spectral and noise patterns typical of AI-generated content.")
        else:
            st.success(f"✅ **REAL IMAGE** | Confidence: {confidence:.2f}%")
            st.info("The image appears consistent with natural camera sensor characteristics.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        os.unlink(tmp_path)

else:
    st.info("Please upload an image to begin analysis.")
