import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model.vqgan import VQGAN
from src.data.factory import MedicalDatasetFactory
from src.analytics.post_process import PostProcessor
from PIL import Image

class AnomalyDetector:
    def __init__(self, config, model_path="checkpoints/vqgan_best.pth"):
        self.config = config
        self.device = torch.device(config['training']['device'])
        
        self.model = VQGAN(
            num_hiddens=config['vqvae']['num_hiddens'],
            num_residual_hiddens=config['vqvae']['num_residual_hiddens'],
            num_embeddings=config['vqvae']['num_embeddings'],
            embedding_dim=config['vqvae']['embedding_dim'],
            commitment_cost=config['vqvae']['commitment_cost']
        ).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"WARNING: Model checkpoint not found at {model_path}")
            
        self.model.eval()
        
        self.transform = MedicalDatasetFactory.get_transforms(config['data']['dataset_type'], config)
        self.post_processor = PostProcessor(sigma=2)

    def detect(self, image_path):
        image = Image.open(image_path).convert('L')
        # Transform returns (C, H, W), we need (B, C, H, W)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # VQVAE forward returns: quantization_loss, reconstruction, perplexity
            _, recon, _ = self.model(input_tensor)
            
            recon_np = recon.squeeze().cpu().numpy()
            input_np = input_tensor.squeeze().cpu().numpy()
            
            # Post-processing
            raw_diff, ssim_map, smoothed_map = self.post_processor.process(input_np, recon_np)
            
            return input_np, recon_np, raw_diff, ssim_map, smoothed_map

    def generate_report_card(self, sick_images, output_path="report_card.png"):
        """
        Generates a visual report card for a list of sick images.
        Shows Original, Reconstruction, Raw Diff, SSIM Map, Smoothed Map.
        """
        n = len(sick_images)
        if n == 0:
            print("No sick images found for report card.")
            return

        # Limit to 5 images
        n = min(n, 5)
        sick_images = sick_images[:n]
        
        plt.figure(figsize=(15, 3*n)) # Adjusted height
        
        for i, img_path in enumerate(sick_images):
            original, recon, raw_diff, ssim_map, smoothed_map = self.detect(img_path)
            
            # Original
            plt.subplot(n, 5, i*5 + 1)
            plt.imshow(original, cmap='gray')
            plt.title("Original")
            plt.axis('off')
            
            # Reconstruction
            plt.subplot(n, 5, i*5 + 2)
            plt.imshow(recon, cmap='gray')
            plt.title("Reconstruction")
            plt.axis('off')
            
            # Raw Difference
            plt.subplot(n, 5, i*5 + 3)
            plt.imshow(raw_diff, cmap='hot')
            plt.title("Raw Diff (L1)")
            plt.axis('off')

            # SSIM Map
            plt.subplot(n, 5, i*5 + 4)
            plt.imshow(ssim_map, cmap='hot')
            plt.title("SSIM Map")
            plt.axis('off')
            
            # Smoothed Map
            plt.subplot(n, 5, i*5 + 5)
            plt.imshow(smoothed_map, cmap='hot')
            plt.title("Smoothed (Final)")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Report card saved to {output_path}")
        plt.close()
