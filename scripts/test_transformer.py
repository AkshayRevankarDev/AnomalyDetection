import torch
import sys
import os
import yaml

# Add root to path
sys.path.append(os.getcwd())

from src.model.transformer import LatentTransformer
from src.model.vqgan import VQGAN

def load_config():
    with open("configs/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def test_transformer():
    print("Testing Latent Transformer...")
    config = load_config()
    device = 'cpu' # Fast test
    
    # 1. Test Forward Pass
    model = LatentTransformer(config['transformer']).to(device)
    idx = torch.randint(0, 1024, (1, 256)).to(device)
    logits, _ = model(idx)
    assert logits.shape == (1, 256, 1024), f"Logits shape mismatch: {logits.shape}"
    print("Forward pass passed.")
    
    # 2. Test Generation
    start_idx = torch.randint(0, 1024, (1, 1)).to(device)
    generated = model.generate(start_idx, max_new_tokens=10)
    assert generated.shape == (1, 256 + 10), f"Generated shape mismatch: {generated.shape}"
    print("Generation passed.")
    
    # 3. Test Decoding (Integration with VQGAN)
    vqgan = VQGAN(
        num_hiddens=config['vqvae']['num_hiddens'],
        num_residual_hiddens=config['vqvae']['num_residual_hiddens'],
        num_embeddings=config['vqvae']['num_embeddings'],
        embedding_dim=config['vqvae']['embedding_dim'],
        commitment_cost=config['vqvae']['commitment_cost']
    ).to(device)
    
    # Create dummy 256 tokens for decoding test
    dummy_generated = torch.randint(0, 1024, (1, 256)).to(device)
    z_q = vqgan.quantizer._embedding(dummy_generated).view(1, 16, 16, 128)
    z_q = z_q.permute(0, 3, 1, 2)
    fake_img = vqgan.decoder(z_q)
    assert fake_img.shape == (1, 1, 256, 256), f"Fake image shape mismatch: {fake_img.shape}"
    print("Decoding passed.")

if __name__ == "__main__":
    try:
        test_transformer()
        print("\nAll Transformer tests passed successfully!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
