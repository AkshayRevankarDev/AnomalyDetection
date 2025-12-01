# MedAnomaly Suite V3.0: The AI Radiologist 🧠

**A Hospital-Grade Neuro-Oncology Dashboard powered by Generative AI.**

This system acts as an intelligent assistant for radiologists, capable of detecting brain tumors, classifying their type, and providing actionable clinical recommendations. It runs locally on Apple Silicon (M-Series) hardware.

## 🚀 Key Features

### 1. High-Fidelity Anomaly Detection
*   **Core**: Vector Quantized Variational Autoencoder (VQ-VAE).
*   **Resolution**: 256x256 (Clinical Grade).
*   **Technique**: Uses **SSIM (Structural Similarity Index)** and **Perceptual Loss** to generate precise anomaly heatmaps, highlighting tumor regions without noise.

### 2. Automated Diagnosis Engine
*   **Core**: ResNet18 Classifier.
*   **Performance**: **99.08% Accuracy**.
*   **Classes**:
    *   No Tumor
    *   Glioma
    *   Meningioma
    *   Pituitary Tumor

### 3. Clinical Logic Module
*   Translates AI predictions into structured reports:
    *   **Risk Level** (High/Medium/Low)
    *   **Immediate Actions** (e.g., "Urgent Oncology Referral")
    *   **Follow-up Protocols**

### 4. Interactive Dashboard
*   Built with **Streamlit**.
*   Provides a side-by-side view of the original scan and the AI-generated heatmap.
*   Displays real-time diagnosis and confidence scores.

---

## 🛠️ Installation & Setup

### Prerequisites
*   Mac with Apple Silicon (M1/M2/M3/M4).
*   Python 3.10+.

### 1. Clone the Repository
```bash
git clone https://github.com/AkshayRevankarDev/AnomalyDetection.git
cd AnomalyDetection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data (Optional for Inference)
The system comes with pre-trained weights (`checkpoints/`). If you want to retrain:
```bash
# Downloads the 4-class Brain MRI dataset from Kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/raw/brain_mri_4class --unzip
```

---

## 🖥️ Usage

### Launch the Dashboard
The easiest way to use the system is via the web interface:

```bash
./run_app.sh
```
This will open `http://localhost:8501` in your browser.

### Manual Inference (CLI)
To run the VQ-VAE anomaly detector on a single image:
```bash
python main.py --mode inference --image path/to/scan.jpg
```

---

## 📂 Project Structure

*   `src/app/`: Streamlit dashboard code.
*   `src/model/`: PyTorch definitions for VQ-VAE and ResNet18.
*   `src/engine/`: Training loops and inference logic.
*   `src/analytics/`: Clinical logic, t-SNE, and post-processing (SSIM).
*   `checkpoints/`: Pre-trained model weights.

---

## 👨‍⚕️ Disclaimer
This tool is for **research and educational purposes only**. It is not a certified medical device and should not be used for primary diagnosis without physician supervision.
