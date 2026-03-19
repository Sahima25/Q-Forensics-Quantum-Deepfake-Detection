import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import pennylane as qml   # not strictly needed here, but kept for consistency

# -------------------------------
# SRM filters (Spatial Rich Model)
# -------------------------------
def get_srm_filters():
    """
    Returns a list of 4 high-pass filter kernels used to extract noise residuals.
    These are simplified versions of the full SRM filter bank.
    """
    f1 = np.array([[0, 0, 0],
                   [1, -2, 1],
                   [0, 0, 0]], dtype=np.float32)
    f2 = np.array([[0, 1, 0],
                   [0, -2, 0],
                   [0, 1, 0]], dtype=np.float32)
    f3 = np.array([[1, 0, 0],
                   [0, -2, 0],
                   [0, 0, 1]], dtype=np.float32)
    f4 = np.array([[0, 0, 1],
                   [0, -2, 0],
                   [1, 0, 0]], dtype=np.float32)
    return [f1, f2, f3, f4]

# -------------------------------
# Noise map extraction (SRM residuals)
# -------------------------------
def extract_noise_map(img_gray, img_size=256):
    """
    Applies SRM filters to a grayscale image and returns the averaged noise map.
    The output is normalized to 0-255 and returned as a uint8 array.
    """
    filters = get_srm_filters()
    residuals = []
    for k in filters:
        res = cv2.filter2D(img_gray, cv2.CV_32F, k)
        residuals.append(res)
    noise = np.mean(residuals, axis=0)
    # Normalize to 0-255 for later conversion to RGB
    noise_norm = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return noise_norm

# -------------------------------
# MobileNetV3 feature extractor
# -------------------------------
# Load the model once (singleton pattern) to avoid reloading
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_mobilenet = None

def get_mobilenet_feature_extractor():
    """
    Returns a MobileNetV3‑Small model without the classifier head,
    moved to the appropriate device.
    """
    global _mobilenet
    if _mobilenet is None:
        print("Loading MobileNetV3‑Small feature extractor...")
        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        except (ImportError, AttributeError):
            # Fallback for older torchvision
            model = models.mobilenet_v3_small(pretrained=True)
        # Keep only the feature extractor (up to and including global average pooling)
        _mobilenet = model.features   # this ends with adaptive avg pool
        _mobilenet = _mobilenet.to(_device)
        _mobilenet.eval()
    return _mobilenet

# Preprocessing pipeline for MobileNetV3
_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_mobilenet_features(img_path):
    """
    Reads an image, computes its SRM noise map, converts to RGB,
    and passes it through MobileNetV3 to obtain a 576‑dimensional feature vector.
    """
    # Read grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (256, 256))

    # Get noise map
    noise_map = extract_noise_map(img)

    # Convert to 3‑channel RGB (by repeating the single channel)
    noise_rgb = np.stack([noise_map] * 3, axis=-1)

    # Preprocess and add batch dimension
    input_tensor = _preprocess(noise_rgb).unsqueeze(0).to(_device)

    # Extract features
    model = get_mobilenet_feature_extractor()
    with torch.no_grad():
        features = model(input_tensor)          # shape (1, 576, 1, 1)
        features = features.view(1, -1).cpu().numpy().flatten()   # (576,)

    return features

# -------------------------------
# Frequency features (FFT)
# -------------------------------
def extract_frequency_features(img_path, img_size=256):
    """
    Reads an image, computes its 2D FFT magnitude spectrum,
    and returns the mean log‑magnitude in low‑ and high‑frequency regions.
    Returns a 2‑element numpy array.
    """
    # Read grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (img_size, img_size))

    # 2D FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) + 1e-8          # avoid log(0)
    log_mag = np.log(mag)

    # Define regions
    center = img_size // 2
    Y, X = np.ogrid[:img_size, :img_size]
    dist = np.sqrt((X - center)**2 + (Y - center)**2)

    low_region = dist < img_size/4
    high_region = dist > img_size/3

    f_low = np.mean(log_mag[low_region])
    f_high = np.mean(log_mag[high_region])

    return np.array([f_low, f_high], dtype=np.float32)

# -------------------------------
# Combined feature extraction (for inference)
# -------------------------------
def extract_all_features(img_path):
    """
    Extracts both MobileNetV3 (576‑dim) and frequency (2‑dim) features,
    concatenates them, and also returns the noise map for visualisation.
    Returns:
        features: torch tensor of shape (578,)
        noise_map: uint8 array (256x256) of SRM residuals (for heatmap)
    """
    # MobileNet features
    mobilenet_feat = extract_mobilenet_features(img_path)          # numpy (576,)

    # Frequency features
    freq_feat = extract_frequency_features(img_path)               # numpy (2,)

    # Concatenate
    combined = np.concatenate([mobilenet_feat, freq_feat])         # (578,)

    # Also get noise map for heatmap (re‑extract to avoid duplication)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (256, 256))
    noise_map = extract_noise_map(img)

    return torch.from_numpy(combined).float(), noise_map
