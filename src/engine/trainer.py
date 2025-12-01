import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from src.model.vqvae import VQVAE
from src.model.perceptual_loss import PerceptualLoss
from src.data.loader import get_data_loaders

class Trainer:
    def __init__(self, config, dry_run=False):
        self.config = config
        self.dry_run = dry_run
        self.device = torch.device(config['training']['device'])
        
        # Initialize Model
        self.model = VQVAE(
            num_hiddens=config['vqvae']['num_hiddens'],
            num_residual_hiddens=config['vqvae']['num_residual_hiddens'],
            num_embeddings=config['vqvae']['num_embeddings'],
            embedding_dim=config['vqvae']['embedding_dim'],
            commitment_cost=config['vqvae']['commitment_cost']
        ).to(self.device)
        
        # Initialize Perceptual Loss
        self.perceptual_loss = PerceptualLoss(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        
        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(config)
        
        self.epochs = config['training']['epochs']
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        train_recon_error = 0
        train_perplexity = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # VQVAE forward returns: quantization_loss, reconstruction, perplexity
            vq_loss, data_recon, perplexity = self.model(data)
            
            # Reconstruction Loss (L1 + Perceptual)
            l1_loss = F.l1_loss(data_recon, data)
            p_loss = self.perceptual_loss(data_recon, data)
            recon_error = l1_loss + 0.1 * p_loss # Weight perceptual loss
            
            # Total Loss
            loss = recon_error + vq_loss
            
            loss.backward()
            train_loss += loss.item()
            train_recon_error += recon_error.item()
            train_perplexity += perplexity.item()
            
            self.optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item(), 'recon': recon_error.item(), 'ppl': perplexity.item()})
            
            if self.dry_run:
                break
        
        avg_loss = train_loss / len(self.train_loader)
        print(f"====> Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        val_recon_error = 0
        val_perplexity = 0
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                vq_loss, data_recon, perplexity = self.model(data)
                l1_loss = F.l1_loss(data_recon, data)
                p_loss = self.perceptual_loss(data_recon, data)
                recon_error = l1_loss + 0.1 * p_loss
                loss = recon_error + vq_loss
                
                val_loss += loss.item()
                val_recon_error += recon_error.item()
                val_perplexity += perplexity.item()
                
                if self.dry_run:
                    break
        
        avg_loss = val_loss / len(self.val_loader)
        print(f"====> Epoch {epoch+1}: Val Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        if self.train_loader is None:
            print("No data found. Skipping training.")
            return

        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), "checkpoints/vqvae_best.pth")
                print(f"Saved best model with val loss: {best_loss:.4f}")
            
            if self.dry_run:
                print("Dry run completed.")
                break
