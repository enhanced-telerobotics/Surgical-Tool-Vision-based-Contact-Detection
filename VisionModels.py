import os
import numpy as np
import json
from datetime import datetime
import random
import torch
import torch.nn as nn
from typing import Optional, List


class CustomCNN(nn.Module):
    def __init__(self,
                 random_seed: Optional[int] = None) -> None:

        super(CustomCNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3),
            nn.ReLU(),
            nn.AvgPool2d(5, stride=2),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(2116, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.states = []

        # Set the random seed if it's provided
        if random_seed:
            os.environ['PYTHONHASHSEED'] = str(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

    def forward(self, X):
        out = self.linear_relu_stack(X)
        return out
