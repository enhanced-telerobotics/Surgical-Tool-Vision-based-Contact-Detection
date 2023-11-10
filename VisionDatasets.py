import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List, Union
from PIL import Image
from utils import JitterCrop


class ContactDataset(Dataset):

    def __init__(self,
                 images: List[object],
                 labels: Union[List[int], np.ndarray],
                 transform: object = torchvision.transforms.Compose([
                     torchvision.transforms.PILToTensor(),
                     torchvision.transforms.ConvertImageDtype(torch.float),
                     torchvision.transforms.RandomRotation(20),
                     JitterCrop((234, 234), 75),
                     torchvision.transforms.RandomHorizontalFlip(),
                     torchvision.transforms.RandomErasing(),
                     torchvision.transforms.ColorJitter(
                         brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
                 ]),
                 weight_coeff: int = 1,
                 finetuning: int = None):

        # Define the transformation pipeline
        if transform is not None:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float),
                JitterCrop((234, 234), 75)
            ])

        # Create a list of indices to shuffle
        if finetuning:
            indices = list(range(len(labels)))
            np.random.shuffle(indices)
            indices = indices[:(round(finetuning * len(labels)))]

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.images = [self.transform(img) for img in images]

        self.distribution = np.bincount(labels)/len(labels)
        self.weightCoeff = weight_coeff

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def getDistribution(self):
        return self.distribution

    def getWeights(self):
        c = self.weightCoeff
        w = self.getDistribution()
        return torch.tensor([c*w[1], c*w[0]], dtype=torch.float)
