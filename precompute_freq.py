import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from feature_extractor import get_model, extract_all_features

# --- CONFIGURATION ---
# Use 'r' before the string to handle Windows backslashes correctly
DATA_ROOT = r"D:\deepfake\data" 
OUTPUT_DIR = "precomputed_features"
LIMIT_PER_CLASS = 5000  # Balanced: 5k Real, 5k Fake

def run_precompute():
    # 1. Setup Folders
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(DATA_ROOT):
        print(f"❌ DATA_ROOT NOT FOUND: {DATA_ROOT}")
        print("Please check if your images are in D:\\deepfake\\data")
        return

    # 2. Initialize the Feature Extractor (MobileNetV3)
    print("⏳ Loading MobileNetV3 Backbone...")
    model = get_model()
    
    # 3. Find Images using Recursive Glob
    # Real images usually sit directly in /real/
    real_pattern = os.path.join(DATA_ROOT, "real", "**", "*.png")
    # Fake images are in /fake/Deepfakes/, /fake/Face2Face/, etc.
    fake_pattern = os.path.join(DATA_ROOT, "fake", "**", "*.png")

    print("🔍 Searching for images...")
    real_paths = glob.glob(real_pattern, recursive=True)
    fake_paths = glob.glob(fake_pattern, recursive=True)

    # Balance the dataset (FaceForensics++ is heavily imbalanced)
    real_selected = real_paths[:LIMIT_PER_CLASS]
    fake_selected = fake_paths[:LIMIT_PER_CLASS]

    tasks = [(p, 0) for p in real_selected] + [(p, 1) for p in fake_selected]
    
    if len(tasks) == 0:
        print(f"❌ No images found!")
        print(f"Checked Real: {real_pattern}")
        print(f"Checked Fake: {fake_pattern}")
        return

    print(f"✅ Found {len(real_selected)} Real and {len(fake_selected)} Fake images.")
    print(f"🚀 Starting Extraction to {OUTPUT_DIR}...")

    # 4. Extraction Loop
    success_count = 0
    for img_path, label in tqdm(tasks):
        try:
            # Extract 578-dim vector (Spatial + Frequency)
            feat = extract_all_features(img_path, model)
            
            if feat is not None:
                # Create a unique filename: folder_filename_label.npy
                parent_dir = os.path.basename(os.path.dirname(img_path))
                file_name = os.path.basename(img_path).split('.')[0]
                save_path = os.path.join(OUTPUT_DIR, f"{parent_dir}_{file_name}_{label}.npy")
                
                np.save(save_path, feat)
                success_count += 1
        except Exception as e:
            # Skip corrupted images
            continue

    print(f"\n✨ SUCCESS: {success_count} feature vectors saved.")
    print(f"Next Step: Run 'python train.py' to start the Quantum GAN training.")

if __name__ == "__main__":
    run_precompute()
