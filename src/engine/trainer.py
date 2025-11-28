import torch
import torch.optim as optim
from tqdm import tqdm
import os
from src.model.vae import VAE
from src.data.loader import get_data_loaders

class Trainer:
    def __init__(self, config, dry_run=False):
        self.config = config
        self.dry_run = dry_run
        self.device = torch.device(config['training']['device'])
        
        # Initialize Model
        self.model = VAE(
            input_channels=config['model']['input_channels'],
            latent_dim=config['model']['latent_dim']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        
        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(config)
        
        self.epochs = config['training']['epochs']
        self.beta = config['training']['beta']
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        train_mse = 0
        train_kld = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch_idx, data in enumerate(progress_bar):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.model(data)
            loss, mse, kld = self.model.loss_function(recon_batch, data, mu, logvar, beta=self.beta)
            
            loss.backward()
            train_loss += loss.item()
            train_mse += mse.item()
            train_kld += kld.item()
            
            self.optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item() / len(data)})
            
            if self.dry_run:
                break
        
        avg_loss = train_loss / len(self.train_loader.dataset)
        print(f"====> Epoch {epoch+1}: Average Train Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss, _, _ = self.model.loss_function(recon_batch, data, mu, logvar, beta=self.beta)
                val_loss += loss.item()
                
                if self.dry_run:
                    break
        
        avg_loss = val_loss / len(self.val_loader.dataset)
        print(f"====> Epoch {epoch+1}: Average Val Loss: {avg_loss:.4f}")
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
                torch.save(self.model.state_dict(), "checkpoints/vae_m4_best.pth")
                print(f"Saved best model with val loss: {best_loss:.4f}")
            
            if self.dry_run:
                print("Dry run completed.")
                break
