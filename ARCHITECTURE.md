# PROJECT: MED-ANOMALY-VAE (Enterprise Edition)
# AUTHOR: Solution Architect
# TARGET HARDWARE: Apple Silicon M4 Pro (Metal Performance Shaders - MPS)

## 1. Executive Summary
Develop a production-grade generative anomaly detection system for medical imaging (Chest X-Rays) using a Variational Autoencoder (VAE). The system will learn the distribution of healthy anatomy and detect pathologies by measuring reconstruction error (MSE + KL Divergence). The solution must be optimized for local training on Apple Silicon (M4 Pro).

## 2. Technical Stack & Constraints
* **Language:** Python 3.10+
* **Framework:** PyTorch (mps-accelerated)
* **Orchestration:** Modular package structure (no monolithic scripts)
* **Visualization:** Matplotlib / Seaborn (for heatmaps)
* **Hardware constraint:** Code must explicitly check for `torch.backends.mps.is_available()` and default to `mps` device.

## 3. Architecture Design
The solution will follow a Clean Architecture pattern with three distinct layers:

### Layer A: Data Ingestion (`src/data`)
* **Dataset:** Kaggle Chest X-Ray (Pneumonia).
* **Pipeline:**
    * Ingest raw JPEG data.
    * Split strategy: TRAIN (Normal Only), VALIDATION (Normal Only), TEST (50% Normal / 50% Sick).
    * Transformations: Resize to 128x128, Grayscale, Min-Max Normalization.
    * Artifact: Custom `PyTorch Dataset` and `DataLoader` classes.

### Layer B: Model Core (`src/model`)
* **Architecture:** Convolutional VAE (Variational Autoencoder).
* **Encoder:** 4x Conv2d Layers (Strided) -> Flatten -> Latent Vector (Z=256).
* **Reparameterization Trick:** Standard Gaussian sampling (Mu + Sigma * Epsilon).
* **Decoder:** 4x ConvTranspose2d Layers -> Sigmoid Activation.
* **Loss Function:** Custom class combining `MSE_Loss` (Reconstruction) + `Beta * KLD_Loss` (Regularization).

### Layer C: Engine (`src/engine`)
* **Trainer:** A dedicated `Trainer` class that handles the training loop, checkpoint saving, and MPS device management.
* **Evaluator:** An `AnomalyDetector` class that:
    1.  Takes an input image.
    2.  Generates reconstruction.
    3.  Computes pixel-wise difference (`abs(input - output)`).
    4.  Generates a heatmap overlay.

## 4. Implementation Plan (Agent Instructions)

### Phase 1: Infrastructure Setup
1.  Create standard python project structure:
    ├── configs/        (YAML config files for hyperparameters)
    ├── data/           (Raw and processed data)
    ├── src/            (Source code)
    │   ├── data/
    │   ├── model/
    │   └── engine/
    ├── notebooks/      (Jupyter/Colab notebooks for analysis)
    └── main.py         (Entry point)
2.  Setup `requirements.txt` with: `torch`, `torchvision`, `matplotlib`, `numpy`, `scikit-learn`, `tqdm`.

### Phase 2: Core Development
1.  Implement `src/model/vae.py`: The VAE class.
2.  Implement `src/data/loader.py`: The data splitting logic. **CRITICAL:** Ensure "Sick" images never leak into the Training set.
3.  Implement `src/engine/trainer.py`: The training loop with `tqdm` progress bars and logging.

### Phase 3: Execution & Artifacts
1.  Train the model for 20 epochs on the M4 Pro.
2.  Save the best weights to `checkpoints/vae_m4_best.pth`.
3.  Generate a "Report Card" image containing:
    * Row 1: 5 Original Sick Images.
    * Row 2: 5 Reconstructed "Healthy" Versions.
    * Row 3: 5 Difference Heatmaps showing the tumor location.

## 5. Definition of Done
* Project runs via `python main.py` without crashing on macOS.
* Training utilizes >80% of GPU (MPS) capacity.
* The system outputs a final anomaly heatmap visualization for visual verification.