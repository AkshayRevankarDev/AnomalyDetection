import torch
import torch.nn as nn
from torchvision import models, transforms

class TumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(TumorClassifier, self).__init__()
        # Load pre-trained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the first layer to accept grayscale images (1 channel)
        # ResNet18 first conv is (3, 64, 7, 7). We change it to (1, 64, 7, 7).
        # We can average the weights of the 3 channels to initialize the 1 channel.
        original_first_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            self.model.conv1.weight.data = original_first_conv.weight.data.mean(dim=1, keepdim=True)
            
        # Modify the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

    @staticmethod
    def get_transforms():
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # No normalization needed if we train from scratch or fine-tune well, 
            # but usually ImageNet mean/std is used. Since we have 1 channel, we can skip or use 0.5.
        ])
