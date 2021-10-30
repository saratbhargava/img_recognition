from typing import Any

from .example_model import ResNet18
from img_recognition.modeling.resnet_model import ResNet


def build_model(cfg) -> Any:
    if cfg.MODEL.NAME == 'resnet18':
        model = ResNet18(cfg.MODEL.NUM_CLASSES)
    elif cfg.MODEL.NAME == 'resnet_n':
        model = ResNet(
            n = cfg.MODEL.NUM_RES_BLOCKS,
            num_classes = cfg.MODEL.NUM_CLASSES)
    return model
