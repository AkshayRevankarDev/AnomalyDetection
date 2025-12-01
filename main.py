import argparse
import yaml
import torch
import os
from src.engine.trainer import Trainer
from src.engine.anomaly_detector import AnomalyDetector

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="VAE Anomaly Detection")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"], help="Mode: train or inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run a single batch for debugging")
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
        trainer = Trainer(config, dry_run=args.dry_run)
        trainer.train()
        
        # Analytics: t-SNE
        from src.analytics.tsne import LatentSpaceVisualizer
        visualizer = LatentSpaceVisualizer(trainer.model, trainer.test_loader, trainer.device)
        visualizer.generate_tsne()
        
        # Analytics: Report Card
        # Filter for sick images in test set
        test_dataset = trainer.test_loader.dataset
        sick_indices = [i for i, label in enumerate(test_dataset.labels) if label == 1]
        sick_paths = [test_dataset.image_paths[i] for i in sick_indices]
        
        if len(sick_paths) > 0:
            # Take first 5 sick images
            sample_sick_paths = sick_paths[:5]
            detector = AnomalyDetector(config, model_path="checkpoints/vqvae_best.pth")
            detector.generate_report_card(sample_sick_paths, output_path="final_report_card.png")
        else:
            print("No sick images found in test set for report card.")
            
    elif args.mode == "inference":
        if not args.image:
            print("Please provide an image path for inference using --image")
            return
        detector = AnomalyDetector(config, model_path="checkpoints/vqvae_best.pth")
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

if __name__ == "__main__":
    main()
