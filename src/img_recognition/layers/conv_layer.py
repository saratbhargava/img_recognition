from torch import nn

class ResNetBlock(nn.Module):
    
    def __init__(self, in_channels: int = 32):
        super().__init__()
        num_filters = in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=num_filters,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters,
                               out_channels=num_filters, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        z = self.conv1(x)
        z = self.relu1(z)
        z = self.bn1(z)
        z = self.conv2(z)
        y = self.relu2(z + x)
        return self.bn2(y)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
