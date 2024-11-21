import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        
        # CNN layers that reduce to 1x1
        self.conv_layers = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 14x14 -> 7x7
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 7x7 -> 3x3
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 3x3 -> 1x1
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=2),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(32, 10)  # Now input is just 32 features from 1x1 spatial dimension
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Will be (batch_size, 32)
        x = self.fc_layers(x)
        return x 