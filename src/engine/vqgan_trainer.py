import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from src.model.vqgan import VQGAN
from src.model.discriminator import NLayerDiscriminator
from src.model.perceptual_loss import PerceptualLoss
from src.model.perceptual_loss import PerceptualLoss
from src.data.loader import get_data_loaders
from src.engine.resilience import CheckpointManager
import time
import datetime
from torchvision.utils import save_image

class VQGANTrainer:
    def __init__(self, config, dry_run=False, time_limit=None):
        self.config = config
        self.dry_run = dry_run
        self.time_limit = time_limit # In minutes
        self.device = torch.device(config['training']['device'])
        
        # Initialize Generator (VQGAN)
        self.vqgan = VQGAN(
            num_hiddens=config['vqvae']['num_hiddens'],
            num_residual_hiddens=config['vqvae']['num_residual_hiddens'],
            num_embeddings=config['vqvae']['num_embeddings'],
            embedding_dim=config['vqvae']['embedding_dim'],
            commitment_cost=config['vqvae']['commitment_cost']
        ).to(self.device)
        
        # Initialize Discriminator
        self.discriminator = NLayerDiscriminator(input_nc=1).to(self.device)
        
        # Initialize Perceptual Loss
        self.perceptual_loss = PerceptualLoss(self.device)
        
        # Optimizers
        lr = config['training']['learning_rate']
        self.opt_vq = optim.Adam(self.vqgan.parameters(), lr=lr, betas=(0.5, 0.9))
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        
        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(config)
        
        self.epochs = config['training']['epochs']
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Resilience
        self.checkpoint_manager = CheckpointManager()

    def train_epoch(self, epoch):
        self.vqgan.train()
        self.discriminator.train()
        
        train_loss_g = 0
        train_loss_d = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch_idx, (imgs, _) in enumerate(progress_bar):
            imgs = imgs.to(self.device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.opt_disc.zero_grad()
            
            # Real images
            pred_real = self.discriminator(imgs)
            loss_d_real = torch.mean((pred_real - 1.0) ** 2) # LSGAN loss
            
            # Fake images
            _, decoded_images, _ = self.vqgan(imgs)
            pred_fake = self.discriminator(decoded_images.detach())
            loss_d_fake = torch.mean(pred_fake ** 2) # LSGAN loss
            
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            self.opt_disc.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            self.opt_vq.zero_grad()
            
            # We need to re-forward to get the computation graph for generator
            vq_loss, decoded_images, perplexity = self.vqgan(imgs)
            
            # GAN Loss
            pred_fake = self.discriminator(decoded_images)
            gan_loss = torch.mean((pred_fake - 1.0) ** 2)
            
            # Reconstruction Loss
            l1_loss = F.l1_loss(decoded_images, imgs)
            p_loss = self.perceptual_loss(decoded_images, imgs)
            recon_loss = l1_loss + 1.0 * p_loss # Weight perceptual loss
            
            # Total Generator Loss
            loss_g = recon_loss + vq_loss + 0.1 * gan_loss
            
            loss_g.backward()
            self.opt_vq.step()
            
            train_loss_g += loss_g.item()
            train_loss_d += loss_d.item()
            
            progress_bar.set_postfix({'g_loss': loss_g.item(), 'd_loss': loss_d.item(), 'ppl': perplexity.item()})
            
            if self.dry_run:
                break
        
        avg_loss_g = train_loss_g / len(self.train_loader)
        avg_loss_d = train_loss_d / len(self.train_loader)
        print(f"====> Epoch {epoch+1}: Avg G Loss: {avg_loss_g:.4f}, Avg D Loss: {avg_loss_d:.4f}")
        return avg_loss_g

    def validate(self, epoch):
        self.vqgan.eval()
        val_loss = 0
        
        with torch.no_grad():
            for imgs, _ in self.val_loader:
                imgs = imgs.to(self.device)
                vq_loss, decoded_images, _ = self.vqgan(imgs)
                l1_loss = F.l1_loss(decoded_images, imgs)
                p_loss = self.perceptual_loss(decoded_images, imgs)
                loss = l1_loss + p_loss + vq_loss
                val_loss += loss.item()
                
                if self.dry_run:
                    break
        
        # Save validation reconstruction samples
        # imgs is the last batch (B, C, H, W)
        # decoded_images is the reconstruction
        # Concatenate them: Original | Recon
        comparison = torch.cat([imgs[:8], decoded_images[:8]])
        save_image(comparison.cpu(), f"outputs/recon_epoch_{epoch+1}.png", nrow=8, normalize=True)
        
        avg_loss = val_loss / len(self.val_loader)
        print(f"====> Epoch {epoch+1}: Val Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        if self.train_loader is None:
            print("No data found. Skipping training.")
            return

        # Auto-Resume
        start_epoch = self.checkpoint_manager.load_checkpoint(
            self.vqgan, self.discriminator, self.opt_vq, self.opt_disc, self.device
        )
        
        best_loss = float('inf')
        start_time = time.time()
        
        print(f"Starting training from epoch {start_epoch+1} to {self.epochs}...")
        
        for epoch in range(start_epoch, self.epochs):
            self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Save best model (Best VQGAN weights only)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.vqgan.state_dict(), "checkpoints/vqgan_best.pth")
                print(f"Saved best model with val loss: {best_loss:.4f}")
            
            # Save Resume Checkpoint (Every epoch)
            self.checkpoint_manager.save_checkpoint(
                epoch, self.vqgan, self.discriminator, self.opt_vq, self.opt_disc, val_loss
            )
            
            # Smart Timeout Check
            if self.time_limit:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes >= self.time_limit:
                    print(f"Time limit of {self.time_limit} minutes reached. Saving and exiting.")
                    break
            
            if self.dry_run:
                print("Dry run completed.")
                break
