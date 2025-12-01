import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        # Load VGG16 pretrained on ImageNet
        vgg = models.vgg16(pretrained=True).features
        
        # Extract features from specific layers to capture different levels of detail
        # ReLU1_2, ReLU2_2, ReLU3_3, ReLU4_3
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
            
        # Freeze parameters (we don't want to train VGG)
        for param in self.parameters():
            param.requires_grad = False
            
        self.to(device)

    def forward(self, recon_x, x):
        # If input is 1 channel (grayscale), repeat to 3 channels for VGG
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            recon_x = recon_x.repeat(1, 3, 1, 1)
            
        h = x
        h_recon = recon_x
        
        h = self.slice1(h)
        h_recon = self.slice1(h_recon)
        loss = torch.mean((h - h_recon) ** 2)
        
        h = self.slice2(h)
        h_recon = self.slice2(h_recon)
        loss += torch.mean((h - h_recon) ** 2)
        
        h = self.slice3(h)
        h_recon = self.slice3(h_recon)
        loss += torch.mean((h - h_recon) ** 2)
        
        h = self.slice4(h)
        h_recon = self.slice4(h_recon)
        loss += torch.mean((h - h_recon) ** 2)
        
        return loss
