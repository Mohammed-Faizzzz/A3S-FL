import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        # Input channels: 3 (for RGB images), Output channels: 32, Kernel size: 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # The input size to the first FC layer is calculated based on the image size and convolutions
        # CIFAR-100 images are 32x32. After three conv/pool layers: 32 -> 16 -> 8 -> 4
        # The number of channels is 128. So, 128 * 4 * 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 100) # Output features: 100 classes for CIFAR-100

    def forward(self, x):
        # Apply conv -> relu -> pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the feature maps for the fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Apply fully connected layers with dropout for regularization
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return x