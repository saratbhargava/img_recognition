import torch
import torch.nn.functional as F
from img_recognition.layers import ResNetBlock, ResNetChangeBlock


def test_ResNetBlock():
    rblock = ResNetBlock(in_channels=64)
    x = torch.randint(0, 100, size=(128, 64, 32, 32), dtype=torch.float32)
    y = rblock(x)
    assert x.shape == y.shape


def test_ResNetChangeBlock():
    rblock2 = ResNetChangeBlock(in_channels=64, num_filters=128)
    x = torch.randint(0, 100, size=(128, 64, 32, 32), dtype=torch.float32)
    y = rblock2(x)
    assert y.shape == (128, 128, 16, 16)


def test_pad_zeros():
    x = torch.ones([128, 16, 16, 16])
    y = torch.ones([128, 32, 16, 16])
    z = F.pad(x, (0, 0, 0, 0, 16, 0, 0, 0))
    assert z.shape == y.shape
