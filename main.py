import argparse
import yaml
import torch
import os
from src.engine.vqgan_trainer import VQGANTrainer
from src.engine.transformer_trainer import TransformerTrainer
from src.engine.anomaly_detector import AnomalyDetector

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="VAE Anomaly Detection")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference", "transformer-train", "dream"], help="Mode: train, inference, transformer-train, dream")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run a single batch for debugging")
    parser.add_argument("--time_limit", type=int, default=None, help="Training time limit in minutes (for auto-resume)")
    parser.add_argument("--image", type=str, help="Path to image for inference")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Ensure MPS is available
    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available. Using CPU. This will be slow.")
        device = "cpu"
    else:
        device = "mps"
    
    print(f"Using device: {device}")
    config['training']['device'] = device

    if args.mode == "train":
        trainer = VQGANTrainer(config, dry_run=args.dry_run, time_limit=args.time_limit)
        trainer.train()
        
        # Analytics: t-SNE
        from src.analytics.tsne import LatentSpaceVisualizer
        visualizer = LatentSpaceVisualizer(trainer.vqgan, trainer.test_loader, trainer.device)
        visualizer.generate_tsne()
        
        # Analytics: Report Card
        # Filter for sick images in test set
        test_dataset = trainer.test_loader.dataset
        sick_indices = [i for i, label in enumerate(test_dataset.labels) if label == 1]
        sick_paths = [test_dataset.image_paths[i] for i in sick_indices]
        
        if len(sick_paths) > 0:
            # Take first 5 sick images
            sample_sick_paths = sick_paths[:5]
            detector = AnomalyDetector(config, model_path="checkpoints/vqgan_best.pth")
            detector.generate_report_card(sample_sick_paths, output_path="final_report_card.png")
        else:
            print("No sick images found in test set for report card.")
            
    elif args.mode == "inference":
        if not args.image:
            print("Please provide an image path for inference using --image")
            return
        detector = AnomalyDetector(config, model_path="checkpoints/vqgan_best.pth")
        original, recon, diff = detector.detect(args.image)
        
        # Save output for verification
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(original, cmap='gray')
        plt.title("Original")
        plt.subplot(1, 3, 2)
        plt.imshow(recon, cmap='gray')
        plt.title("Reconstruction")
        plt.subplot(1, 3, 3)
        plt.imshow(diff, cmap='hot')
        plt.title("Difference")
        plt.savefig("inference_result.png")
        print("Inference result saved to inference_result.png")

    elif args.mode == "transformer-train":
        trainer = TransformerTrainer(config, dry_run=args.dry_run, time_limit=args.time_limit)
        trainer.train()
        
    elif args.mode == "dream":
        # Dream Mode: Generate a synthetic patient
        from src.model.vqgan import VQGAN
        from src.model.transformer import LatentTransformer
        import matplotlib.pyplot as plt
        
        # Load VQGAN
        vqgan = VQGAN(
            num_hiddens=config['vqvae']['num_hiddens'],
            num_residual_hiddens=config['vqvae']['num_residual_hiddens'],
            num_embeddings=config['vqvae']['num_embeddings'],
            embedding_dim=config['vqvae']['embedding_dim'],
            commitment_cost=config['vqvae']['commitment_cost']
        ).to(device)
        vqgan.load_state_dict(torch.load("checkpoints/vqgan_best.pth", map_location=device))
        vqgan.eval()
        
        # Load Transformer
        transformer = LatentTransformer(config['transformer']).to(device)
        if os.path.exists("checkpoints/transformer_best.pth"):
            transformer.load_state_dict(torch.load("checkpoints/transformer_best.pth", map_location=device))
        else:
            print("WARNING: Transformer checkpoint not found. Generating noise.")
        transformer.eval()
        
        # Generate
        print("Dreaming of a new patient...")
        # Start with a random token
        idx = torch.randint(0, config['transformer']['vocab_size'], (1, 1)).to(device)
        
        # Generate 255 more tokens (total 256 for 16x16)
        with torch.no_grad():
            generated_indices = transformer.generate(idx, max_new_tokens=255, temperature=1.0)
            
            # Decode with VQGAN
            z_q = vqgan.quantizer._embedding(generated_indices).view(1, 16, 16, 128) # (B, H, W, C)
            z_q = z_q.permute(0, 3, 1, 2) # (B, C, H, W)
            
            fake_img = vqgan.decoder(z_q)
            
            # Save
            plt.imsave("dream_patient.png", fake_img.squeeze().cpu().numpy(), cmap='gray')
            print("Dream saved to dream_patient.png")

if __name__ == "__main__":
    main()
