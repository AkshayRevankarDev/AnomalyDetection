# VAE Anomaly Detection for Medical Imaging

This project implements a Variational Autoencoder (VAE) for anomaly detection in Chest X-Rays, optimized for Apple Silicon (MPS).

## Prerequisites

- Python 3.10+
- Kaggle Account (for dataset)

## Setup

1.  **Clone the repository**
    ```bash
    git clone <your-repo-url>
    cd Patern_recognition
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Setup**
    You need the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle.
    
    **Option A: Automatic Download (Recommended)**
    1.  Get your Kaggle API credentials:
        - Go to Kaggle Settings -> API -> Create New Token.
        - This downloads `kaggle.json`.
    2.  Place `kaggle.json` in `~/.kaggle/kaggle.json` OR set environment variables:
        ```bash
        export KAGGLE_USERNAME=your_username
        export KAGGLE_KEY=your_key
        ```
    3.  Run the setup script:
        ```bash
        python scripts/download_data.py
        ```

    **Option B: Manual Download**
    1.  Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
    2.  Extract to `data/raw` so the structure is:
        ```
        data/raw/chest_xray/train/...
        data/raw/chest_xray/test/...
        ```

## Usage

### Training
To train the model (default 20 epochs):
```bash
python main.py --mode train
```
Checkpoints are saved to `checkpoints/`.

### Inference
To detect anomalies in a specific image:
```bash
python main.py --mode inference --image path/to/image.jpeg
```
This generates `inference_result.png`.

## Project Structure
- `src/`: Source code (model, data loader, trainer).
- `configs/`: Configuration files.
- `scripts/`: Helper scripts.
- `data/`: Data directory (ignored by git).
