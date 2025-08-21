import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_features=3, num_classes=100):
        super().__init__()
        
        # --- Conv Block 1 ---
        self.conv1 = nn.Conv2d(in_features, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # --- Conv Block 2 ---
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # --- Conv Block 3 ---
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # --- Fully connected ---
        self.fc1 = nn.Linear(256 * 4 * 4, 512)   # after 3x maxpool -> 32 -> 16 -> 8 -> 4
        self.fc2 = nn.Linear(512, num_classes)
        
        # --- Layers ---
        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.act(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Conv block 2
        x = self.act(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        
        # Conv block 3
        x = self.act(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        
        return x
