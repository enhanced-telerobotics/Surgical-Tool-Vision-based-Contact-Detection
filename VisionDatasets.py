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
    """
    A dataset class for handling image data with optional finetuning, jitter effects, and label weighting.

    This class extends the PyTorch Dataset and allows for a subset of data to be selected based on a finetuning parameter.
    It supports image transformations including jitter effects and coordinates-based cropping.

    Attributes:
        labels (torch.Tensor): Tensor of labels corresponding to the images.
        coords (List[Tuple[int, int]]): List of coordinates for image cropping, if provided.
        images (List[torch.Tensor]): List of image tensors.
        distribution (numpy.ndarray): Distribution of the labels in the dataset.
        weightCoeff (int): Coefficient for weighting the labels in loss calculation.
        jitter (bool): Flag to apply jitter transformations to the images.

    Parameters:
        images (List[object]): List of image file paths or objects.
        labels (Union[List[int], np.ndarray]): List or array of integer labels corresponding to the images.
        coords (Optional[List[Tuple[int, int]]]): List of (y, x) tuples for center cropping of images. Defaults to None.
        jitter (bool): If True, apply random jitter transformations to the images. Defaults to False.
        weight_coeff (int): Coefficient used for label weighting. Defaults to 1.
        finetuning (Optional[float]): A float in the range (0, 1] indicating the proportion of data to use. Defaults to None.

    Methods:
        __len__: Returns the number of items in the dataset.
        __getitem__(index): Retrieves the image and label at the specified index after applying transformations.
        getDistribution(): Returns the distribution of labels in the dataset.
        getWeights(): Calculates and returns the weights for each class based on the distribution and weight coefficient.
    """

    def __init__(self,
                 images: List[object],
                 labels: Union[List[int], np.ndarray],
                 coords: List[Tuple[int, int]] = None,
                 jitter: bool = False,
                 weight_coeff: int = 1,
                 finetuning: Optional[float] = None):

        if finetuning is not None:
            assert 0 < finetuning <= 1, "finetuning must be in the range (0, 1]"

            indices = np.random.permutation(len(images))
            subset_size = int(len(images) * finetuning)
            selected_indices = indices[:subset_size]

            images = [images[i] for i in selected_indices]
            labels = [labels[i] for i in selected_indices]
            if coords:
                coords = [coords[i] for i in selected_indices]

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.coords = coords
        self.images = []

        for image in images:
            image = Image.open(image)
            image = torchvision.transforms.PILToTensor()(image)
            self.images.append(image)
        self.images = torch.stack(self.images, dim=0)

        self.distribution = np.bincount(labels)/len(labels)
        self.weightCoeff = weight_coeff
        self.jitter = jitter

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.coords:
            cy, cx = self.coords[index]
            jitter = 10
        else:
            cx, cy = None, None
            jitter = 75

        # determine if add jitter transforms
        if self.jitter:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(20),
                CenterCrop((234, 234), jitter, cx, cy),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomErasing(),
                torchvision.transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]
            )
        else:
            self.transform = torchvision.transforms.Compose([
                CenterCrop((234, 234), jitter, cx, cy),
                torchvision.transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]
            )

        image = self.images[index]
        image = self.transform(image)
        image = image.float()

        return image, self.labels[index]

    def getDistribution(self):
        return self.distribution

    def getWeights(self):
        c = self.weightCoeff
        w = self.getDistribution()
        return torch.tensor([c*w[1], c*w[0]], dtype=torch.float)
