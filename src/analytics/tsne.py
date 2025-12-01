import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

class LatentSpaceVisualizer:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def generate_tsne(self, output_path="outputs/tsne_plot.png"):
        self.model.eval()
        all_latents = []
        all_labels = []
        
        print("Extracting latent vectors for t-SNE...")
        with torch.no_grad():
            for data, labels in self.dataloader:
                data = data.to(self.device)
                
                # Get quantized latents
                # z = encoder(x) -> pre_quant -> quantizer
                z = self.model.encoder(data)
                z = self.model.pre_quantization_conv(z)
                _, quantized, _, _ = self.model.quantizer(z)
                
                # Flatten: (B, C, H, W) -> (B, C*H*W)
                # C=64, H=8, W=8 -> 4096 dim vector
                # Reduce dimensionality for t-SNE stability: Average pooling -> (B, C)
                latents = torch.mean(quantized, dim=[2, 3]).cpu().numpy()
                
                all_latents.append(latents)
                all_labels.append(labels.numpy())
                
        all_latents = np.concatenate(all_latents, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Map labels to strings for legend
        label_map = {0: 'Healthy', 1: 'Pathology'}
        str_labels = [label_map[l] for l in all_labels]
        
        print(f"Running t-SNE on {all_latents.shape[0]} vectors of dim {all_latents.shape[1]}...")
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_latents)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=tsne_results[:, 0], y=tsne_results[:, 1],
            hue=str_labels,
            palette={'Healthy': 'blue', 'Pathology': 'red'},
            alpha=0.6
        )
        
        plt.title('t-SNE Visualization of VQ-VAE Latent Space')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"t-SNE plot saved to {output_path}")
