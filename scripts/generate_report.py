import sys
import os
import yaml
import glob
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.anomaly_detector import AnomalyDetector

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config_path = "configs/config.yaml"
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return

    config = load_config(config_path)
    
    # Ensure MPS is available (or use CPU for inference if needed)
    import torch
    if not torch.backends.mps.is_available():
        device = "cpu"
    else:
        device = "mps"
    config['training']['device'] = device

    print("Initializing Anomaly Detector...")
    try:
        detector = AnomalyDetector(config)
    except FileNotFoundError:
        print("Error: Model checkpoint not found. Please train the model first.")
        return

    # Find Sick Images
    raw_path = config['data']['raw_path']
    # Look for PNEUMONIA images in test or val folders, or just generally in raw
    sick_images = glob.glob(os.path.join(raw_path, "**", "PNEUMONIA", "*.jpeg"), recursive=True) + \
                  glob.glob(os.path.join(raw_path, "**", "PNEUMONIA", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(raw_path, "**", "PNEUMONIA", "*.png"), recursive=True)
    
    if len(sick_images) < 5:
        print(f"Not enough sick images found. Found {len(sick_images)}")
        return

    # Select 5 random images
    selected_images = random.sample(sick_images, 5)
    
    print("Generating Report Card...")
    detector.generate_report_card(selected_images, output_path="final_report_card.png")
    print("Done!")

if __name__ == "__main__":
    main()
