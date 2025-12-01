import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from src.model.classifier import TumorClassifier

class ClassifierTrainer:
    def __init__(self, data_path, device="mps", batch_size=32, epochs=10, learning_rate=1e-4):
        self.device = torch.device(device)
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.model = TumorClassifier(num_classes=4).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.train_loader, self.val_loader, self.classes = self._get_dataloaders()
        
    def _get_dataloaders(self):
        # Data Augmentation and Normalization
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        
        # Assuming data_path has 'Training' and 'Testing' folders or similar structure
        # The Kaggle dataset usually has 'Training' and 'Testing'
        train_dir = os.path.join(self.data_path, 'Training')
        test_dir = os.path.join(self.data_path, 'Testing')
        
        if not os.path.exists(train_dir):
            # Fallback if structure is different (e.g. flat folders)
            # We will use ImageFolder on the root and split
            full_dataset = datasets.ImageFolder(self.data_path, transform=transform)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
            classes = full_dataset.classes
        else:
            train_dataset = datasets.ImageFolder(train_dir, transform=transform)
            val_dataset = datasets.ImageFolder(test_dir, transform=transform)
            classes = train_dataset.classes
            
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, classes

    def train(self):
        print(f"Classes found: {self.classes}")
        best_acc = 0.0
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': running_loss/total, 'acc': 100*correct/total})
                
            # Validation
            val_acc = self.validate()
            print(f"Epoch {epoch+1} Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "checkpoints/diagnosis_resnet.pth")
                print("Saved best model.")
                
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
