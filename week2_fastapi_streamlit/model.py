import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolution block 1
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=8, 
            kernel_size=3, 
            padding=1
        )  # → output: 8 x 28 x 28

        self.pool = nn.MaxPool2d(2, 2)  # → output becomes 8 x 14 x 14

        # Convolution block 2
        self.conv2 = nn.Conv2d(
            in_channels=8, 
            out_channels=16, 
            kernel_size=3, 
            padding=1
        )  # → output: 16 x 14 x 14, pooled → 16 x 7 x 7

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
