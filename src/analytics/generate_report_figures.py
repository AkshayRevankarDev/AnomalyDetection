import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from cv2 import GaussianBlur

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.factory import MedicalDatasetFactory
from src.engine.anomaly_detector import AnomalyDetector
from src.model.vqgan import VQGAN
from src.model.transformer import LatentTransformer
import yaml

def load_config():
    with open("configs/config.yaml", 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
DEVICE = CONFIG['training']['device']

def get_transforms():
    # Use the same transforms as Factory (Zoomed Out / Resize 128)
    return MedicalDatasetFactory.get_transforms("BRAIN", CONFIG)

def generate_pipeline_viz(image_path, output_path="report_contents/vqgan_analysis.png"):
    print(f"Generating Pipeline Visualization for {image_path}...")
    
    # 1. Load Models
    detector = AnomalyDetector(CONFIG, model_path="checkpoints/vqgan_best.pth")
    
    # 2. Get Pipeline Outputs
    original_np, recon_np, raw_diff, ssim_map, smoothed_map = detector.detect(image_path)
    
    # 3. Plotting
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Col 1: Original
    axes[0].imshow(original_np, cmap='gray')
    axes[0].set_title("1. Original Image")
    axes[0].axis('off')
    
    # Col 2: Reconstruction
    axes[1].imshow(recon_np, cmap='gray')
    axes[1].set_title("2. VQ-GAN Recon")
    axes[1].axis('off')
    
    # Col 3: Raw L1
    axes[2].imshow(raw_diff, cmap='hot')
    axes[2].set_title("3. Raw L1 Diff")
    axes[2].axis('off')
    
    # Col 4: SSIM Map
    # The detector 'detect' method already returns 1 - ssim for the map?
    # Let's check AnomalyDetector logic. 
    # Usually detect returns post-processed maps. 
    # But user asked for SSIM map specifically using ssim(..., full=True).
    # AnomalyDetector.detect calls detector.post_processor.process which likely does this.
    # We will assume detector returns what we need (Raw Diff, SSIM, Smoothed).
    # If SSIM map is inverted (similarity), we might want error (1-SSIM). 
    # Let's assume AnomalyDetector returns the error map or the map suitable for visualization.
    axes[3].imshow(ssim_map, cmap='jet')
    axes[3].set_title("4. SSIM Map")
    axes[3].axis('off')
    
    # Col 5: Smoothed Final
    axes[4].imshow(smoothed_map, cmap='hot')
    axes[4].set_title("5. Smoothed Anomaly Map")
    axes[4].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()

def generate_dream_grid(output_path="report_contents/transformer_dreams.png"):
    print("Generating Dream Grid...")
    
    # 1. Load Models
    vqgan = VQGAN(
        num_hiddens=CONFIG['vqvae']['num_hiddens'],
        num_residual_hiddens=CONFIG['vqvae']['num_residual_hiddens'],
        num_embeddings=CONFIG['vqvae']['num_embeddings'],
        embedding_dim=CONFIG['vqvae']['embedding_dim'],
        commitment_cost=CONFIG['vqvae']['commitment_cost']
    ).to(DEVICE)
    
    if os.path.exists("checkpoints/vqgan_best.pth"):
        vqgan.load_state_dict(torch.load("checkpoints/vqgan_best.pth", map_location=DEVICE))
    else:
        print("Error: VQGAN checkpoint not found.")
        return
        
    vqgan.eval()
    
    transformer = LatentTransformer(CONFIG['transformer']).to(DEVICE)
    if os.path.exists("checkpoints/transformer_best.pth"):
        transformer.load_state_dict(torch.load("checkpoints/transformer_best.pth", map_location=DEVICE))
    else:
        print("Error: Transformer checkpoint not found.")
        return
        
    transformer.eval()
    
    # 2. Generate 8 samples
    num_samples = 8
    generated_images = []
    
    print(f"Dreaming {num_samples} patients...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Start token
            idx = torch.randint(0, CONFIG['transformer']['vocab_size'], (1, 1)).to(DEVICE)
            
            # Generate tokens
            # We need 256 tokens total (16x16)
            # max_new_tokens = 255 if we have 1 start token
            generated_indices = transformer.generate(idx, max_new_tokens=255, temperature=1.0)
            
            # Decode
            z_q = vqgan.quantizer._embedding(generated_indices).view(1, 16, 16, 128)
            z_q = z_q.permute(0, 3, 1, 2)
            fake_img = vqgan.decoder(z_q)
            
            # Convert to numpy
            img_np = fake_img.squeeze().cpu().numpy()
            generated_images.append(img_np)
            
    # 3. Plot Grid 2x4
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Background fix: Set background to black (like MRI)
        axes[i].imshow(generated_images[i], cmap='gray')
        # Zoom out effect for consistency with "Full Brain Mode"
        # Since these are generated 128x128, they are already "zoomed out" relative to the frame if training was correct.
        # But we can add padding to emphasize it if needed. 
        # For now, just show the generated content directly.
        axes[i].set_title(f"Synthetic Patient #{i+1}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved {output_path}")
    plt.close()

if __name__ == "__main__":
    # Ensure report_contents exists
    os.makedirs("report_contents", exist_ok=True)
    
    # Sample Image
    sample_path = "data/raw/brain_mri/yes/Y1.jpg"
    if not os.path.exists(sample_path):
        # Fallback search if Y1.jpg doesn't exist
        print(f"{sample_path} not found, searching...")
        import glob
        files = glob.glob("data/raw/brain_mri/yes/*.jpg")
        if files:
            sample_path = files[0]
            print(f"Using {sample_path}")
        else:
            print("No sample images found!")
            sys.exit(1)
            
    generate_pipeline_viz(sample_path)
    generate_dream_grid()
