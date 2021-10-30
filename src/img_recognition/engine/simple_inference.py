import logging

from tqdm import tqdm
from yacs.config import CfgNode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from img_recognition.engine import DATASET

logger = logging.getLogger(__name__)


def test_fn(
    config: CfgNode,
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.modules.loss._Loss,
    dataset: DATASET,
) -> bool:

    device = config.MODEL.DEVICE

    loss = 0
    results = []
    model.eval()
    with torch.no_grad():
        for batch_index, (X, y) in tqdm(enumerate(test_dataloader)):
            X_data, y_data = X.to(device), y.to(device)
            y_pred = model(X_data)
            loss += loss_fn(y_pred, y_data).item()
            results.extend(torch.argmax(y_pred, dim=-1) == y_data)

    test_accu = sum(results) / len(results)
    test_accu = test_accu.item()
    test_loss = loss / len(results)

    logger.info(f"{dataset} loss: {test_loss:.4f}, {dataset} accuracy: {test_accu:.4f}")

    return False
