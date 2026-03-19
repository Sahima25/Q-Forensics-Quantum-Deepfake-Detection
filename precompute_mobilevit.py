import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from tqdm import tqdm
import csv
import argparse

# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(description='Precompute MobileNetV3 features from SRM noise maps.')
parser.add_argument('--data_root', type=str, default='data',
                    help='Root folder containing real/ and fake/ subfolders.')
parser.add_argument('--output_dir', type=str, default='mobilevit_features',
                    help='Directory where features and mapping will be saved.')
parser.add_argument('--max_per_class', type=int, default=5000,
                    help='Maximum number of images per class (for balanced subset). Use -1 for all.')
parser.add_argument('--img_size', type=int, default=256,
                    help='Image size for resizing.')
args = parser.parse_args()

data_root = args.data_root
output_dir = args.output_dir
max_per_class = args.max_per_class
img_size = args.img_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Load MobileNetV3 feature extractor
# -------------------------------
print("Loading MobileNetV3‑Small feature extractor...")
try:
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
except (ImportError, AttributeError):
    base_model = models.mobilenet_v3_small(pretrained=True)

# Keep only the feature extractor (includes global average pooling)
base_model = base_model.features
base_model = base_model.to(device)
base_model.eval()

# Feature dimension is fixed at 576 for MobileNetV3‑Small
feature_dim = 576
print(f"Feature dimension: {feature_dim}")

# -------------------------------
# Preprocessing pipeline
# -------------------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------
# SRM filters
# -------------------------------
def get_srm_filters():
    f1 = np.array([[0,0,0],[1,-2,1],[0,0,0]], dtype=np.float32)
    f2 = np.array([[0,1,0],[0,-2,0],[0,1,0]], dtype=np.float32)
    f3 = np.array([[1,0,0],[0,-2,0],[0,0,1]], dtype=np.float32)
    f4 = np.array([[0,0,1],[0,-2,0],[1,0,0]], dtype=np.float32)
    return [f1, f2, f3, f4]

filters = get_srm_filters()

def extract_noise_map(img_gray):
    residuals = []
    for k in filters:
        res = cv2.filter2D(img_gray, cv2.CV_32F, k)
        residuals.append(res)
    noise = np.mean(residuals, axis=0)
    noise_norm = cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return noise_norm

# -------------------------------
# Gather all image paths
# -------------------------------
image_paths = []
labels = []
for class_name in ['real', 'fake']:
    class_dir = os.path.join(data_root, class_name)
    if not os.path.isdir(class_dir):
        print(f"Warning: {class_dir} not found. Skipping.")
        continue
    for root, _, files in os.walk(class_dir):
        for f in files:
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                image_paths.append(os.path.join(root, f))
                labels.append(0 if class_name == 'real' else 1)

print(f"Total images found: {len(image_paths)} (Real: {labels.count(0)}, Fake: {labels.count(1)})")

# -------------------------------
# Limit to balanced subset
# -------------------------------
if max_per_class > 0:
    real_paths = [p for p, l in zip(image_paths, labels) if l == 0][:max_per_class]
    fake_paths = [p for p, l in zip(image_paths, labels) if l == 1][:max_per_class]
    image_paths = real_paths + fake_paths
    labels = [0]*len(real_paths) + [1]*len(fake_paths)
    print(f"Using subset: {len(real_paths)} real, {len(fake_paths)} fake (total {len(image_paths)})")

# -------------------------------
# CSV mapping file
# -------------------------------
csv_path = os.path.join(output_dir, 'paths.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['base_name', 'original_path'])

    # Process each image
    for img_path in tqdm(image_paths, desc="Extracting features"):
        try:
            # Read grayscale image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: cannot read {img_path}")
                continue
            img = cv2.resize(img, (img_size, img_size))

            # Get noise map
            noise_map = extract_noise_map(img)

            # Convert to 3‑channel RGB
            noise_rgb = np.stack([noise_map]*3, axis=-1)

            # Preprocess and run through model
            input_tensor = preprocess(noise_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                features = base_model(input_tensor)          # (1,576,1,1)
                features = features.view(1, -1).cpu().numpy().flatten()

            # Generate base name (relative path with underscores)
            rel_path = os.path.relpath(img_path, data_root)
            base_name = rel_path.replace(os.sep, '_')

            # Save features
            np.save(os.path.join(output_dir, base_name + '.npy'), features)

            # Write mapping
            writer.writerow([base_name, img_path])

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print(f"Done! Features saved to {output_dir}")
print(f"Mapping saved to {csv_path}")
