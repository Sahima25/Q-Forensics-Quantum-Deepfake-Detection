# Q-Forensics-Quantum-Deepfake-Detection
Unmasking synthetic media through 4-qubit entanglement and 16-D Hilbert Space auditing.
## Overview:
 Q-Forensics is a lightweight hybrid quantum-classical framework designed to address the growing threat of high-fidelity synthetic media. While classical deep learning models require millions of parameters to detect advanced generative artifacts (from models like Flux and Stable Diffusion 3.5), Q-Forensics leverages Quantum Machine Learning to achieve high-precision detection with a fraction of the computational overhead.

 ## Key Features
### Quantum Advantage: 
Utilizes a 4-Qubit Variational Quantum Circuit (VQC) to achieve 97.8% accuracy.
### Hilbert Space Mapping : 
Features are mapped into a 16-dimensional ($2^4$) Hilbert Space, allowing for superior separation of real vs. fake data points.
### Dual-Stream Fusion: 
Processes both Spatial Noise Residuals (SRM) and Frequency Domain Artifacts (2D-FFT).
### Provenance Auditing: 
Beyond detection, it verifies digital watermarks and AI company logos to establish a clear content origin.
### Efficiency: 
 80% reduction in parameters compared to traditional CNNs, optimized for a ~45ms inference time.

## Technical Architecture & Decision Logic
### Forensic Feature Extraction: 
Decomposes images into "checkerboard artifacts" (Frequency) and pixel-level noise residuals (Spatial).

### Quantum Embedding: 
Maps these features into a 8-dimensional Hilbert Space using Angle Embedding.

### Entanglement Analysis: 
A 4-qubit VQC uses CNOT gates to find correlations between artifacts that are invisible to classical detectors.

## Final Verdict & Explainability: 
  ### Binary Classification: 
  Determines the probability of the media being Real or Synthetic.

  ### JET Heatmap Generation: 
  Highlights "areas of interest" where the manipulation occurred, providing human-readable proof.

 ### Provenance Audit: 
  Cross-references results with detected AI watermarks or logos for a final "Forensic Veto."

 
## 🚀 Getting Started

Follow these instructions to set up and run Q‑Forensics on your own machine.

### 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- (Optional) A GPU with CUDA support for faster training

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sahima25/Q-Forensics-Quantum-Deepfake-Detection.git
   cd Q-Forensics-Quantum-Deepfake-Detection
2.Install the required packages:

bash
pip install -r requirements.txt
The main dependencies are:

torch and torchvision – for deep learning models

pennylane – for quantum circuit simulation

streamlit – for the web interface

opencv-python, numpy, tqdm – for image processing and utilities
Data Preparation
Your dataset should be organised in the following structure:

text
data/
├── real/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── fake/
    ├── fake1.jpg
    ├── fake2.png
    └── ...
Place your real and fake face images in the corresponding folders. For best results, use a balanced dataset (e.g., 5000 real + 5000 fake).

 Precomputation of Features
Before training, you need to extract the spatial (MobileNetV3) and frequency (FFT) features for all images. Run these two scripts in order:

MobileNetV3 features from SRM noise maps
This script processes images, applies SRM filters, and extracts a 576‑dimensional feature vector using a pretrained MobileNetV3‑Small.

bash
python precompute_mobilevit.py --data_root data --output_dir mobilevit_features --max_per_class 5000
--max_per_class 5000 creates a balanced subset of 5000 real and 5000 fake images.

The script also saves a paths.csv file that maps base names to original image paths.

Frequency features (FFT)
This script reads the mapping from the previous step and extracts low‑ and high‑frequency FFT means for the same images.

bash
python precompute_freq.py --mapping_csv mobilevit_features/paths.csv --output_dir precomputed_features_subset
After this, you will have two folders:

mobilevit_features/ – contains .npy files (576‑dim each)

precomputed_features_subset/ – contains .pt files (2‑dim each)

 Training the Hybrid Model
Once the features are precomputed, you can train the hybrid quantum‑classical model:

bash
python train_hybrid_concat.py --mobilevit_dir mobilevit_features --freq_dir precomputed_features_subset --epochs 20
Training takes about 3–4 hours on a CPU (faster on GPU).

The model weights will be saved in weights/hybrid_concat_weights.pth.

 Running the Streamlit App
After training, you can test the model with a user‑friendly web interface:

bash
streamlit run app.py
Then open your browser at the URL shown (usually http://localhost:8501). Upload an image and see the forensic analysis, including the heatmap and the quantum‑based verdict.

📊 Results
On a balanced subset of 10,000 images (5,000 real + 5,000 fake), our hybrid model achieves ~97.9% validation accuracy, outperforming a purely classical baseline (92%). The quantum circuit adds measurable value by capturing correlations invisible to classical methods.

 
