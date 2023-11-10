import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import math
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List


def train(model: nn.Module,
          optimizer: optim.Optimizer,
          loss_fn: nn.Module,
          device: torch.device,
          dataloader: DataLoader,
          val_dataloader: Optional[DataLoader] = None,
          epochs: int = 100,
          use_tqdm: bool = True):

    if use_tqdm:
        from tqdm import tqdm
        pbar = tqdm(total=epochs*len(dataloader),
                    desc="Training Progress")

    model.to(device)
    for epoch in range(epochs):
        correct = 0
        train_loss = 0.0

        for X, y in dataloader:
            model.train()
            # Move data to the device
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Accumulate loss and count correct predictions using tensor operations
            train_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_tqdm:
                pbar.set_description(f"Epoch {epoch}")
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

        # Calculate average loss and accuracy over the entire dataset
        train_loss /= len(dataloader)
        train_acc = correct / len(dataloader.dataset)

        model.metrics['train_loss'].append(train_loss)
        model.metrics['train_acc'].append(train_acc)

        # If a validation dataloader is provided
        if val_dataloader:
            model.eval()
            val_correct = 0
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in val_dataloader:
                    X_val, y_val = X_val.to(device), y_val.to(device)

                    pred_val = model(X_val)
                    val_loss += loss_fn(pred_val, y_val).item()
                    val_correct += (pred_val.argmax(1) ==
                                    y_val).type(torch.float).sum().item()

            val_loss /= len(val_dataloader)
            val_acc = val_correct / len(val_dataloader.dataset)

            model.metrics['val_loss'].append(val_loss)
            model.metrics['val_acc'].append(val_acc)

        model.states.append(model.state_dict())

    # add smooth windows for all metrics
    for key in model.metrics:
        if len(model.metrics[key]) > 0:
            model.metrics[key] = _moving_average(model.metrics[key]).tolist()

    return model.metrics


def predict(model: nn.Module,
            dataloader: DataLoader,
            device: torch.device) -> List[torch.Tensor]:
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            output = model(X)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().detach().numpy())

    return np.array(predictions)


def _moving_average(data: np.ndarray,
                    window_size: int = 5):
    """ Compute moving average using numpy. """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


class CenterCrop(object):
    """
    Crop the input image around a specified center with random jitter.

    Args:
        output_size (tuple): Desired output size (height, width) of the crop.
        vec_len (int): Maximum jitter vector length.
        cx (int, optional): Center x-coordinate for cropping. If None, the image center is used.
        cy (int, optional): Center y-coordinate for cropping. If None, the image center is used.

    Returns:
        PIL.Image: Cropped image.

    Example:
        >>> crop_transform = CenterCrop(output_size=(100, 100), vec_len=10, cx=50, cy=50)
        >>> cropped_image = crop_transform(input_image)
    """

    def __init__(self, output_size, vec_len=75, cx=None, cy=None):
        """
        Initialize the CenterCrop transform.

        Args:
            output_size (tuple): Desired output size (height, width) of the crop.
            vec_len (int): Maximum jitter vector length.
            cx (int, optional): Center x-coordinate for cropping. If None, the image center is used.
            cy (int, optional): Center y-coordinate for cropping. If None, the image center is used.
        """
        self.size = output_size
        self.len = vec_len
        self.cx = cx
        self.cy = cy

    def __call__(self, im):
        """
        Crop the input image.

        Args:
            im (PIL.Image): Input image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        height, weight = im.shape[1:3]

        # Calculate the crop center coordinates
        if self.cx is None:
            cx = math.ceil(weight / 2)
            diff_x = 0
        else:
            cx = self.cx
            diff_x = int(math.ceil(self.len * random.uniform(-1, 1)))

        if self.cy is None:
            cy = math.ceil(height / 2)
            diff_y = int(math.ceil(self.len * random.uniform(-1, 1)))
        else:
            cy = self.cy
            diff_y = 0

        # Calculate the coordinates of the top-left corner of the crop
        top = cx - self.size[0] // 2 + diff_x
        left = cy - self.size[1] // 2 + diff_y

        # Use torchvision's crop function for cropping
        return torchvision.transforms.functional.crop(im, left, top, self.size[0], self.size[1])
