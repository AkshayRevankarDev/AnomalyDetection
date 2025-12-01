import torch
import os
import logging

class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints", checkpoint_name="vqgan_resume.pth"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, epoch, vqgan, discriminator, opt_vq, opt_disc, loss):
        """Saves the training state."""
        state = {
            'epoch': epoch,
            'vqgan_state_dict': vqgan.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'opt_vq_state_dict': opt_vq.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict(),
            'loss': loss
        }
        try:
            torch.save(state, self.checkpoint_path)
            print(f"Checkpoint saved to {self.checkpoint_path} (Epoch {epoch})")
        except Exception as e:
            print(f"ERROR: Failed to save checkpoint: {e}")

    def load_checkpoint(self, vqgan, discriminator, opt_vq, opt_disc, device):
        """Loads the training state if it exists. Returns start_epoch."""
        if not os.path.exists(self.checkpoint_path):
            print("No resume checkpoint found. Starting from scratch.")
            return 0

        try:
            print(f"Loading checkpoint from {self.checkpoint_path}...")
            checkpoint = torch.load(self.checkpoint_path, map_location=device)
            
            vqgan.load_state_dict(checkpoint['vqgan_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            opt_vq.load_state_dict(checkpoint['opt_vq_state_dict'])
            opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1 # Resume from next epoch
            print(f"Resumed training from Epoch {start_epoch}")
            return start_epoch
            
        except Exception as e:
            print(f"ERROR: Corrupted checkpoint found at {self.checkpoint_path}. Starting from scratch. Error: {e}")
            return 0
