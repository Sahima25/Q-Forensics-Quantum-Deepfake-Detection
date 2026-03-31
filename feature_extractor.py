import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    features = model.features
    extractor = nn.Sequential(features, nn.AdaptiveAvgPool2d(1)).to(device)
    extractor.eval()
    return extractor

def get_srm_noise(img_gray):
    # Standard SRM filters for consistency
    f1 = np.array([[0,0,0],[1,-2,1],[0,0,0]], dtype=np.float32)
    f2 = np.array([[0,1,0],[0,-2,0],[0,1,0]], dtype=np.float32)
    f3 = np.array([[1,0,0],[0,-2,0],[0,0,1]], dtype=np.float32)
    res = [cv2.filter2D(img_gray, cv2.CV_32F, k) for k in [f1, f2, f3]]
    noise = np.mean(res, axis=0)
    return cv2.normalize(noise, None, 0, 1, cv2.NORM_MINMAX)

def extract_features_from_array(img_gray, model):
    """New function to ensure app.py and train.py are identical"""
    img_resized = cv2.resize(img_gray, (256, 256))
    
    # SRM Noise Logic
    noise_map = get_srm_noise(img_resized)
    noise_rgb = cv2.merge([noise_map, noise_map, noise_map])
    tensor = transforms.ToTensor()(noise_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        spatial_feat = model(tensor).view(-1).cpu().numpy()

    # Frequency Logic
    f = np.fft.fft2(img_resized)
    mag = np.abs(np.fft.fftshift(f)) + 1e-8
    log_mag = np.log(mag)
    freq_feat = np.array([np.mean(log_mag), np.std(log_mag)], dtype=np.float32)
    
    return np.concatenate([spatial_feat, freq_feat])

def extract_all_features(img_path, model):
    """Legacy wrapper for file paths"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    return extract_features_from_array(img, model)
