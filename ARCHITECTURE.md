# PROJECT: MED-ANOMALY-SUITE-ULTIMATE (v5.3)
# TYPE: Research-Grade Medical AI Platform
# TARGET HARDWARE: Apple Silicon M4 Pro (MPS Optimized)

## 1. Executive Summary
A full-stack "AI Radiologist" platform combining unsupervised anomaly detection (VQ-GAN), explainable diagnosis (ResNet + Grad-CAM), and generative data synthesis (Latent Transformer).

## 2. System Architecture Layers

### Layer A: Data Ingestion (`src/data`)
* **Transformation:** `Resize((128, 128))` -> `Grayscale` -> `ToTensor`. (Full Skull Mode).

### Layer B: The Visual Core (`src/model/vqgan.py`)
* **Architecture:** VQ-GAN (Vector Quantized GAN).
* **Output:** High-fidelity reconstruction.

### Layer C: The Diagnostic Brain (`src/model/classifier.py`)
* **Architecture:** ResNet18 + Grad-CAM.

### Layer D: The Generative Engine (`src/model/transformer.py`)
* **Architecture:** MinGPT (Latent Transformer).
* **Capability:** "Dream Mode" (Synthetic Data Generation).

### Layer G: Reporting & Visualization (`src/analytics`)
* **Module:** `generate_report_figures.py`
    * **Figure A (Detailed VQ-GAN Analysis):** A 5-column plot for a single patient:
        1. **Original Scan** (Input).
        2. **Reconstruction** (VQ-GAN Output).
        3. **Raw L1 Difference** (Pixel-wise absolute error).
        4. **SSIM Map** (Structural Similarity error map).
        5. **Smoothed Anomaly Map** (Final heatmap used for detection).
    * **Figure B (Transformer Dreams):** A grid of synthesized patients generated from pure noise.

## 3. Implementation Priorities (Immediate)

### Priority 1: Generate Visual Evidence
* Run `src/analytics/generate_report_figures.py`.
* Calculate accurate SSIM maps using `scikit-image`.
* Save all outputs to `report contents/` at 300 DPI.