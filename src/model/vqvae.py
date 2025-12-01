import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import VectorQuantizer

class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        
        # Encoder
        # Input: (B, 1, 128, 128)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_hiddens // 2, kernel_size=4, stride=2, padding=1), # -> (B, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1), # -> (B, 128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1), # -> (B, 128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(num_hiddens, num_residual_hiddens, kernel_size=4, stride=2, padding=1), # -> (B, 32, 16, 16)
            nn.ReLU(),
             nn.Conv2d(num_residual_hiddens, num_residual_hiddens, kernel_size=4, stride=2, padding=1), # -> (B, 32, 8, 8)
            nn.ReLU(),
        )
        
        self.pre_quantization_conv = nn.Conv2d(num_residual_hiddens, embedding_dim, kernel_size=1, stride=1)
        
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, num_residual_hiddens, kernel_size=4, stride=2, padding=1), # -> (B, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(num_residual_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1), # -> (B, 128, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1), # -> (B, 128, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1), # -> (B, 64, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(num_hiddens // 2, 1, kernel_size=4, stride=2, padding=1), # -> (B, 1, 128, 128)
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_quantization_conv(z)
        loss, quantized, perplexity, _ = self.quantizer(z)
        x_recon = self.decoder(quantized)
        
        return loss, x_recon, perplexity
