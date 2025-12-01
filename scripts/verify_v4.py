import torch
import sys
import os
import yaml
import numpy as np
from PIL import Image

# Add root to path
sys.path.append(os.getcwd())

from src.engine.vqgan_trainer import VQGANTrainer
from src.model.classifier import TumorClassifier
from src.analytics.gradcam import GradCAM

def load_config():
    with open("configs/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def test_vqgan_training():
    print("Testing VQGAN Training Loop...")
    config = load_config()
    # Force dry run and CPU for quick test if MPS not available (but user has MPS)
    # config['training']['device'] = 'cpu' 
    trainer = VQGANTrainer(config, dry_run=True)
    trainer.train()
    print("VQGAN Training Loop Passed!")

def test_gradcam():
    print("Testing Grad-CAM...")
    model = TumorClassifier(num_classes=4)
    model.eval()
    
    # Create dummy image (1, 128, 128)
    dummy_input = torch.randn(1, 1, 128, 128)
    
    gradcam = GradCAM(model.model, model.model.layer4)
    heatmap = gradcam(dummy_input)
    
    assert heatmap.shape == (128, 128), f"Heatmap shape mismatch: {heatmap.shape}"
    print("Grad-CAM Passed!")

if __name__ == "__main__":
    try:
        test_vqgan_training()
        test_gradcam()
        print("\nAll V4.0 verification tests passed successfully!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
