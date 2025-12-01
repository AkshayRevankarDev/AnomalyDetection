import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
from src.model.vqgan import VQGAN
from src.model.transformer import LatentTransformer
from src.data.loader import get_data_loaders

class TransformerTrainer:
    def __init__(self, config, dry_run=False, time_limit=None):
        self.config = config
        self.dry_run = dry_run
        self.time_limit = time_limit
        self.device = torch.device(config['training']['device'])
        
        # Load VQGAN (Frozen)
        self.vqgan = VQGAN(
            num_hiddens=config['vqvae']['num_hiddens'],
            num_residual_hiddens=config['vqvae']['num_residual_hiddens'],
            num_embeddings=config['vqvae']['num_embeddings'],
            embedding_dim=config['vqvae']['embedding_dim'],
            commitment_cost=config['vqvae']['commitment_cost']
        ).to(self.device)
        
        # Load weights
        if os.path.exists("checkpoints/vqgan_best.pth"):
            self.vqgan.load_state_dict(torch.load("checkpoints/vqgan_best.pth", map_location=self.device))
            print("Loaded VQGAN for Transformer training.")
        else:
            print("WARNING: VQGAN checkpoint not found! Transformer will learn random noise.")
            
        self.vqgan.eval()
        for param in self.vqgan.parameters():
            param.requires_grad = False
            
        # Initialize Transformer
        self.model = LatentTransformer(config['transformer']).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['transformer']['learning_rate'])
        
        # Data Loaders
        self.train_loader, self.val_loader, _ = get_data_loaders(config)
        
        self.epochs = config['transformer']['epochs']
        os.makedirs("checkpoints", exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Transformer]")
        
        for batch_idx, (imgs, _) in enumerate(progress_bar):
            imgs = imgs.to(self.device)
            
            # 1. Get Latent Indices from VQGAN
            with torch.no_grad():
                z = self.vqgan.encoder(imgs)
                z = self.vqgan.pre_quantization_conv(z)
                _, _, _, encoding_indices = self.vqgan.quantizer(z)
                
                # Reshape indices to (B, T)
                # encoding_indices is (B*H*W, 1) -> (B, H*W)
                indices = encoding_indices.view(imgs.shape[0], -1)
                
            # 2. Prepare Inputs and Targets
            # Input: indices[:, :-1]
            # Target: indices[:, 1:]
            # But wait, MinGPT usually trains on the whole sequence to predict the next token at each step.
            # forward(idx, targets) handles this internally if we pass the same sequence as target?
            # No, standard GPT training:
            # x = indices[:, :-1]
            # y = indices[:, 1:]
            
            # Actually, let's look at our forward pass:
            # logits, loss = self.forward(idx, targets)
            # It computes cross_entropy(logits, targets)
            # So we should pass full sequence as input and target?
            # No, usually x is input, y is target shifted by 1.
            
            x = indices[:, :-1]
            y = indices[:, 1:]
            
            self.optimizer.zero_grad()
            logits, loss = self.model(x, y)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            if self.dry_run:
                break
                
        avg_loss = train_loss / len(self.train_loader)
        print(f"====> Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        if self.train_loader is None:
            return

        best_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.epochs):
            loss = self.train_epoch(epoch)
            
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), "checkpoints/transformer_best.pth")
                print(f"Saved best transformer with loss: {best_loss:.4f}")
                
            if self.time_limit:
                elapsed = (time.time() - start_time) / 60
                if elapsed >= self.time_limit:
                    print("Time limit reached.")
                    break
                    
            if self.dry_run:
                break
