import os
import numpy as np
import random
import torch
import torch.nn as nn
from typing import Optional, List
from torchvision.models import efficientnet_b3


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

        if random_seed:
            os.environ['PYTHONHASHSEED'] = str(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

    def forward(self, X):
        return self.linear_relu_stack(X)


class CustomEfficientNetB3(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 pretrained: bool = True,
                 random_seed: Optional[int] = None):
        super(CustomEfficientNetB3, self).__init__()
        self.base_model = efficientnet_b3(pretrained=pretrained)

        self.dropout = nn.Dropout(p=0.3, inplace=True)
        self.fc = nn.Linear(
            self.base_model.classifier[1].in_features, num_classes)

        # Replace the classifier of the base model
        self.base_model.classifier = nn.Sequential(
            self.dropout,
            self.fc
        )

        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.states = []

        if random_seed:
            os.environ['PYTHONHASHSEED'] = str(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)

    def forward(self, x):
        return self.base_model(x)
