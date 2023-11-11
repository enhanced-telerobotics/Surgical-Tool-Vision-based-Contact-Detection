import math
import random
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List, Union, Tuple
from PIL import Image
from utils import CenterCrop


class ContactDataset(Dataset):

    def __init__(self,
                 images: List[object],
                 labels: Union[List[int], np.ndarray],
                 coords: List[Tuple[int, int]],
                 jitter: bool = False,
                 weight_coeff: int = 1,
                 finetuning: Optional[int] = None):
        
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.coords = coords
        self.images = images

        self.distribution = np.bincount(labels)/len(labels)
        self.weightCoeff = weight_coeff
        self.jitter = jitter

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        cx, cy = self.coords[index]

        # determine if add jitter transforms
        if self.jitter:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(20),
                CenterCrop((234, 234), 10, cx, cy),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomErasing(),
                torchvision.transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]
            )
        else:
            self.transform = torchvision.transforms.Compose([
                CenterCrop((234, 234), 10, cx, cy),
                torchvision.transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]
            )
        
        image = self.images[index]
        image = Image.open(image)
        image = torchvision.transforms.PILToTensor()(image)
        image = self.transform(image)
        image = image.float()

        return image, self.labels[index]

    def getDistribution(self):
        return self.distribution

    def getWeights(self):
        c = self.weightCoeff
        w = self.getDistribution()
        return torch.tensor([c*w[1], c*w[0]], dtype=torch.float)

