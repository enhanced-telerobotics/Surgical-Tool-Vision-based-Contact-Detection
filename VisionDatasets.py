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
                 transform: List[object] = None,
                 jitter: bool = False,
                 weight_coeff: int = 1,
                 finetuning: Optional[int] = None,
                 coords: Optional[Tuple[int, int]] = None):

        # Create a list of indices to shuffle
        if finetuning:
            indices = list(range(len(labels)))
            np.random.shuffle(indices)
            indices = indices[:(round(finetuning * len(labels)))]

        self.labels = torch.tensor(labels, dtype=torch.long)

        # Define the transformation pipeline
        if transform is None:
            self.transform = [
                torchvision.transforms.PILToTensor()
            ]
        else:
            self.transform = transform

        # add default crop
        crop_size = (234, 234)
        self.transform.append(CenterCrop(crop_size))

        # determine if add jitter transforms
        if jitter:
            self.transform.extend([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomErasing(),
                torchvision.transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]
            )

        # set center of crop if provided
        if coords is None:
            transform_compose = torchvision.transforms.Compose(self.transform)
            self.images = [transform_compose(img).float() for img in images]
        else:
            self.images = []
            for ((cx, cy), img) in zip(coords, images):
                if jitter:
                    self.transform[-4] = CenterCrop(crop_size, cx=cx, cy=cy)
                else:
                    self.transform[-1] = CenterCrop(crop_size, cx=cx, cy=cy)
                transform_compose = torchvision.transforms.Compose(
                    self.transform)
                self.images.append(transform_compose(img).float())

        # calc distribution for weights
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
