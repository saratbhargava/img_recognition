import logging

from tqdm import tqdm
from yacs.config import CfgNode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_fn(
    config: CfgNode,
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
) -> None:

    device = config.MODEL.DEVICE

    model.train()

    for X, y in tqdm(train_dataloader, total=len(train_dataloader)):

        X_data, y_data = X.to(device), y.to(device)
        y_pred = model(X_data)
        loss = loss_fn(y_pred, y_data)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = loss.item()
    accu = torch.argmax(y_pred, dim=-1) == y_data
    accu = sum(accu) / len(accu)
    accu = accu.item()
    logger.info(f"Train loss: {loss:.4f}, Train accu: {accu:.4f}")
