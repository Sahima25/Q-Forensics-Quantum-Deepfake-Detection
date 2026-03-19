import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import csv
import argparse

# -------------------------------
# Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(description='Precompute frequency features (FFT) for images listed in a mapping CSV.')
parser.add_argument('--mapping_csv', type=str, default='mobilevit_features/paths.csv',
                    help='Path to the CSV file containing base_name and original_path.')
parser.add_argument('--output_dir', type=str, default='precomputed_features_subset',
                    help='Directory where frequency .pt files will be saved.')
parser.add_argument('--img_size', type=int, default=256,
                    help='Image size for resizing.')
args = parser.parse_args()

mapping_csv = args.mapping_csv
output_dir = args.output_dir
img_size = args.img_size

os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Load mapping
# -------------------------------
base_to_path = {}
with open(mapping_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        base_to_path[row['base_name']] = row['original_path']

print(f"Loaded {len(base_to_path)} image paths from {mapping_csv}")

# -------------------------------
# Frequency feature extraction
# -------------------------------
def extract_frequency_features(img_gray):
    """Return low and high frequency means as a 2‑element tensor."""
    img = cv2.resize(img_gray, (img_size, img_size))
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) + 1e-8
    log_mag = np.log(mag)

    center = img_size // 2
    Y, X = np.ogrid[:img_size, :img_size]
    dist = np.sqrt((X - center)**2 + (Y - center)**2)

    low_region = dist < img_size/4
    high_region = dist > img_size/3

    f_low = np.mean(log_mag[low_region])
    f_high = np.mean(log_mag[high_region])

    return torch.tensor([f_low, f_high], dtype=torch.float32)

# -------------------------------
# Process each image
# -------------------------------
success = 0
for base_name, img_path in tqdm(base_to_path.items(), desc="Extracting frequency features"):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: cannot read {img_path}")
            continue
        freq_feat = extract_frequency_features(img)
        out_path = os.path.join(output_dir, base_name + '.pt')
        torch.save(freq_feat, out_path)
        success += 1
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print(f"Successfully saved {success} frequency feature files to {output_dir}")
