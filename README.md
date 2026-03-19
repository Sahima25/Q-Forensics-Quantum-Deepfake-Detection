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
