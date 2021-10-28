import torch
from img_recognition.modeling.resnet_model import ResNet


def test_resnet_model():
    n = 3
    num_classes = 10
    batch_size = 128
    res_net = ResNet(n=n, num_classes=num_classes)
    x = torch.randint(0, 100, size=(batch_size, 3, 32, 32), dtype=torch.float32)
    y = res_net(x)
    assert y.shape == (batch_size, num_classes)