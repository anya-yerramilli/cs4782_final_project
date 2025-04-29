import torch
import torch.nn as nn
import torch.nn.functional as F


######################################
############### Conv-4 ###############
######################################

# Define basic convolutional block
class ConvBlock(nn.Module):
  """
  Builds basic convolutional block. See link for a detailed description: https://arxiv.org/pdf/1703.05175#page=5.
  """
  def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2)
    )

  def forward(self, x):
    return self.model(x)

# Define Conv-4
class Conv4(nn.Module):
  def __init__(self):
    super(Conv4, self).__init__()

    self.model = nn.Sequential(
      ConvBlock(3, 64),
      ConvBlock(64, 64),
      ConvBlock(64, 64),
      ConvBlock(64, 64)
    )

  def forward(self, x):
    return self.model(x)


#####################################################
############### ResNet-10 & ResNet-18 ###############
#####################################################


# Define a residual block
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    # shortcut connection to match dimensions if needed
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
      )

  def forward(self, x):
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = self.relu(out)
    return out

# Define ResNet-10
class ResNet10(nn.Module):
  def __init__(self):
    super(ResNet10, self).__init__()
    self.in_channels = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)

    self.layer1 = self._make_layer(64, 1, stride=1)
    self.layer2 = self._make_layer(128, 1, stride=2)
    self.layer3 = self._make_layer(256, 1, stride=2)
    self.layer4 = self._make_layer(512, 1, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

  def _make_layer(self, out_channels, blocks, stride):
    layers = []
    layers.append(ResidualBlock(self.in_channels, out_channels, stride))
    self.in_channels = out_channels
    for _ in range(1, blocks):
      layers.append(ResidualBlock(self.in_channels, out_channels))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    return x

# Define ResNet-18
class ResNet18(nn.Module):
  """
  Buils 18-layer residual network. See link for a detailed description: https://arxiv.org/pdf/1703.05175.
  """
  def __init__(self):
    super(ResNet18, self).__init__()
    self.in_channels = 64

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)

    self.layer1 = self._make_layer(64, 2, stride=1)
    self.layer2 = self._make_layer(128, 2, stride=2)
    self.layer3 = self._make_layer(256, 2, stride=2)
    self.layer4 = self._make_layer(512, 2, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

  def _make_layer(self, out_channels, blocks, stride):
    layers = []
    layers.append(ResidualBlock(self.in_channels, out_channels, stride))
    self.in_channels = out_channels
    for _ in range(1, blocks):
      layers.append(ResidualBlock(self.in_channels, out_channels))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    return x


###################################
############### WRN ###############
###################################


# Define WRN
class WRN(nn.Module):
  def __init__(self, depth : int = 28, widen_factor : int = 10):
    super(WRN, self).__init__()

    assert (depth - 4) % 6 == 0, 'depth not of the form 6n+4'
    n = (depth - 4) // 6  # number of blocks per layer

    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.layer1 = self._make_layer(16, 16 * widen_factor, n, stride=1)
    self.layer2 = self._make_layer(16 * widen_factor, 32 * widen_factor, n, stride=2)
    self.layer3 = self._make_layer(32 * widen_factor, 64 * widen_factor, n, stride=2)
    
    self.bn2 = nn.BatchNorm2d(64 * widen_factor)

  def _make_layer(self, in_channels, out_channels, num_blocks, stride):
    layers = []
    layers.append(ResidualBlock(in_channels, out_channels, stride))
    for _ in range(1, num_blocks):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.relu(self.bn2(out))
    out = F.avg_pool2d(out, 8)
    return out


#########################################
############### MobileNet ###############
#########################################


class MobileBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super(MobileBlock, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                padding=1, groups=in_channels, bias=False),
      nn.BatchNorm2d(in_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.model(x)

class MobileNet(nn.Module):
  def __init__(self):
    super(MobileNet, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),

      MobileBlock(32, 64, stride=1),
      MobileBlock(64, 128, stride=2),
      MobileBlock(128, 128, stride=1),
      MobileBlock(128, 256, stride=2),
      MobileBlock(256, 256, stride=1),
      MobileBlock(256, 512, stride=2),
      *[MobileBlock(512, 512, stride=1) for _ in range(5)],
      MobileBlock(512, 1024, stride=2),
      MobileBlock(1024, 1024, stride=1),

      nn.AdaptiveAvgPool2d(1)
    )

  def forward(self, x):
    return self.model(x)