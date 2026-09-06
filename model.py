import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path


logger = logging.getLogger(Path(__file__).name)


# region LeNet-5 model architecture
class LeNet5(nn.Module):
    """
    Example model to demonstrate distributed data parallelism (DDP)
    """
    def __init__(self, n_classes: int = 10, in_channels: int = 1):
        super(LeNet5, self).__init__()

        # Define convolutional layers
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.c3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        # Define pooling layers
        self.s1 = nn.AvgPool2d(kernel_size=2)
        self.s2 = nn.AvgPool2d(kernel_size=2)

        # Define the fully connected layers fc1 and fc2
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.s1(F.tanh(self.c1(x)))
        x = self.s2(F.tanh(self.c2(x)))
        x = F.tanh(self.c3(x))
        x = x.view(batch_size, -1)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def reset_parameters(self):
        """
        Reinitialize model parameters with Kaiming uniform initialization, and zero for the bias.
        """
        for layer in [self.c1, self.c2, self.c3, self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="tanh")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
# endregion
