import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import random
from sklearn.model_selection import train_test_split

class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L') # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros((1, 128, 128)), 0 # Return dummy image and label on error

        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx] if self.labels is not None else 0
        
        return image, label

from .factory import MedicalDatasetFactory

def get_data_loaders(config):
    raw_path = config['data']['raw_path']
    batch_size = config['training']['batch_size']
    dataset_type = config['data'].get('dataset_type', 'LUNG')
    
    # 1. Get Transforms and Paths via Factory
    transform = MedicalDatasetFactory.get_transforms(dataset_type, config)
    normal_images, sick_images = MedicalDatasetFactory.get_paths(dataset_type, raw_path)

    print(f"[{dataset_type}] Found {len(normal_images)} Normal images and {len(sick_images)} Sick images.")

    if len(normal_images) == 0:
        print("WARNING: No images found. Please check data path.")
        return None, None, None

    # 2. Split Strategy
    # TRAIN: Normal Only
    # VALIDATION: Normal Only
    # TEST: Mixed (Normal + Sick)
    
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    # test_split is remaining
    
    # Split Normal images: Train + Val + Test_Normal
    # We want Train to be pure normal.
    # Let's use sklearn train_test_split
    
    # First split off the Test set portion of Normal images
    # If train=0.8, val=0.1, then test=0.1
    test_size = config['data']['test_split']
    
    train_val_normal, test_normal = train_test_split(normal_images, test_size=test_size, random_state=42)
    
    # Now split Train and Val
    # relative val size = val_split / (train_split + val_split)
    relative_val_size = val_split / (train_split + val_split)
    
    train_normal, val_normal = train_test_split(train_val_normal, test_size=relative_val_size, random_state=42)
    
    # Test set includes the reserved normal images AND all sick images
    test_paths = test_normal + sick_images
    test_labels = [0] * len(test_normal) + [1] * len(sick_images)
    
    train_labels = [0] * len(train_normal)
    val_labels = [0] * len(val_normal)
    
    train_dataset = ChestXRayDataset(train_normal, labels=train_labels, transform=transform)
    val_dataset = ChestXRayDataset(val_normal, labels=val_labels, transform=transform)
    test_dataset = ChestXRayDataset(test_paths, labels=test_labels, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for safety on some systems, can bump to 2
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
