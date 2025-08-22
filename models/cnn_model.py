import torch
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # Load pretrained ResNet18 but adjust input/output for CIFAR-100
        self.model = models.resnet18(weights=None)  # or weights="IMAGENET1K_V1" if you want pretrained
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # CIFAR images are small (32x32), so skip this
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
