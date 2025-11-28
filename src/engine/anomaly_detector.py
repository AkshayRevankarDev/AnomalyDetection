import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model.vae import VAE
from torchvision import transforms
from PIL import Image

class AnomalyDetector:
    def __init__(self, config, model_path="checkpoints/vae_m4_best.pth"):
        self.config = config
        self.device = torch.device(config['training']['device'])
        
        self.model = VAE(
            input_channels=config['model']['input_channels'],
            latent_dim=config['model']['latent_dim']
        ).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"WARNING: Model checkpoint not found at {model_path}")
            
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((config['model']['input_height'], config['model']['input_width'])),
            transforms.ToTensor()
        ])

    def detect(self, image_path):
        image = Image.open(image_path).convert('L')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reconstruction, _, _ = self.model(input_tensor)
            
        input_np = input_tensor.cpu().squeeze().numpy()
        recon_np = reconstruction.cpu().squeeze().numpy()
        
        # Compute difference
        diff = np.abs(input_np - recon_np)
        
        return input_np, recon_np, diff

    def generate_report_card(self, sick_images, output_path="report_card.png"):
        n = len(sick_images)
        plt.figure(figsize=(15, 9))
        
        for i, img_path in enumerate(sick_images):
            original, recon, diff = self.detect(img_path)
            
            # Original
            plt.subplot(3, n, i + 1)
            plt.imshow(original, cmap='gray')
            plt.title("Original")
            plt.axis('off')
            
            # Reconstruction
            plt.subplot(3, n, i + 1 + n)
            plt.imshow(recon, cmap='gray')
            plt.title("Reconstruction")
            plt.axis('off')
            
            # Heatmap
            plt.subplot(3, n, i + 1 + 2*n)
            plt.imshow(diff, cmap='hot')
            plt.title("Difference")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Report card saved to {output_path}")
