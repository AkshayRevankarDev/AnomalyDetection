import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=256):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (B, 1, 128, 128)
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), # -> (B, 32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> (B, 64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (B, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> (B, 256, 8, 8)
            nn.ReLU()
        )
        
        self.flatten_dim = 256 * 8 * 8
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            # Input: (B, 256, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> (B, 128, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> (B, 64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> (B, 32, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # -> (B, 1, 128, 128)
            nn.Sigmoid() # Pixel values between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x_encoded = self.encoder(x)
        x_flat = torch.flatten(x_encoded, start_dim=1)
        
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        z_projected = self.decoder_input(z)
        z_reshaped = z_projected.view(-1, 256, 8, 8)
        reconstruction = self.decoder(z_reshaped)
        
        return reconstruction, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        # Reconstruction loss (MSE)
        # Sum over all pixels
        MSE = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL Divergence
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return MSE + beta * KLD, MSE, KLD
