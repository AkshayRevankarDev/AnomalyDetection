import torch
import sys
import os
import yaml
import time
import shutil

# Add root to path
sys.path.append(os.getcwd())

from src.engine.vqgan_trainer import VQGANTrainer
from src.engine.resilience import CheckpointManager

def load_config():
    with open("configs/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def test_checkpoint_manager():
    print("Testing CheckpointManager...")
    # Clean up previous tests
    if os.path.exists("checkpoints/test_resume.pth"):
        os.remove("checkpoints/test_resume.pth")
        
    manager = CheckpointManager(checkpoint_name="test_resume.pth")
    
    # Mock objects
    vqgan = torch.nn.Linear(10, 10)
    discriminator = torch.nn.Linear(10, 10)
    opt_vq = torch.optim.Adam(vqgan.parameters())
    opt_disc = torch.optim.Adam(discriminator.parameters())
    
    # Save
    manager.save_checkpoint(5, vqgan, discriminator, opt_vq, opt_disc, 0.5)
    assert os.path.exists("checkpoints/test_resume.pth")
    
    # Load
    start_epoch = manager.load_checkpoint(vqgan, discriminator, opt_vq, opt_disc, 'cpu')
    assert start_epoch == 6 # Should be epoch + 1
    print("CheckpointManager Passed!")
    
    # Clean up
    if os.path.exists("checkpoints/test_resume.pth"):
        os.remove("checkpoints/test_resume.pth")

if __name__ == "__main__":
    try:
        test_checkpoint_manager()
        print("\nAll Resilience tests passed successfully!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
