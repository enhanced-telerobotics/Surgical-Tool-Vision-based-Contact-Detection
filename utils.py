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


class JitterCrop(object):
    def __init__(self, output_size, vec_len):
        self.size = output_size
        self.len = vec_len

    def __call__(self, im):
        x, y = im.shape[1:3]
        x = math.ceil(-self.size[0]/2 + x/2)
        y = math.ceil(-self.size[1]/2 + y/2)
        dx = int(math.ceil(self.len*random.uniform(-1, 1)))
        dy = int(math.ceil(self.len*random.uniform(-1, 1)))
        return torchvision.transforms.functional.crop(im, x+dx, y+dy, self.size[0], self.size[1])
