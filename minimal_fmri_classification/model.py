import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################################
# Basic 3D Residual Blocks
#################################################################

class BasicBlock3D(nn.Module):
    """
    A standard 3D BasicBlock (like ResNet-18/34 in 3D).
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class BottleneckBlock3D(nn.Module):
    """
    A 3D Bottleneck block (similar to ResNet-50/101/152 in 3D).
    Expands channels by a factor of 4.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = F.relu(out, inplace=True)
        return out


#################################################################
# 3D ResNet Definition
#################################################################

class ResNet3D(nn.Module):
    """
    A configurable 3D ResNet. The `block` can be `BasicBlock3D`
    (ResNet-18/34 style) or `BottleneckBlock3D` (ResNet-50+ style).
    The `layers` list specifies the number of blocks in each of the 4 layers.
    """

    def __init__(self, block, layers, in_channels=1, n_classes=1, base_channels=64):
        super().__init__()
        self.in_channels = base_channels

        # Initial stem
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, base_channels,   layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels*8, layers[3], stride=2)

        # Classification head
        out_channels = base_channels*8 * block.expansion
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(out_channels, n_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        layers = []
        # First block in this layer may change channels or do stride > 1
        layers.append(block(self.in_channels, out_channels, stride=stride))
        # If it's a bottleneck, out_channels is effectively out_channels * 4
        if hasattr(block, "expansion"):
            self.in_channels = out_channels * block.expansion
        else:
            self.in_channels = out_channels

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, self.in_channels, stride=1))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Main layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pool + FC
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


#################################################################
# Predefined ResNet configs (including a smaller custom "resnet10")
#################################################################

RESNET_CONFIGS = {
    "resnet10":  {"block": BasicBlock3D,      "layers": [1, 1, 1, 1]},  # Custom "tiny" version
    "resnet18":  {"block": BasicBlock3D,      "layers": [2, 2, 2, 2]},
    "resnet34":  {"block": BasicBlock3D,      "layers": [3, 4, 6, 3]},
    "resnet50":  {"block": BottleneckBlock3D, "layers": [3, 4, 6, 3]},
    "resnet101": {"block": BottleneckBlock3D, "layers": [3, 4, 23, 3]},
    "resnet152": {"block": BottleneckBlock3D, "layers": [3, 8, 36, 3]},
}

#################################################################
# Factory function: build_model
#################################################################

def build_model(config):
    """
    Usage example:
        config = {
            "model": {
                "variant": "resnet10",
                "in_channels": 1,
                "n_classes": 1,
                "base_channels": 64,
            }
        }
    """
    variant = config["model"]["variant"]
    in_channels = config["model"]["in_channels"]
    n_classes = config["model"]["n_classes"]
    base_channels = config["model"]["base_channels"]

    if variant not in RESNET_CONFIGS:
        raise ValueError(f"Unknown 3D ResNet variant: {variant}")

    block  = RESNET_CONFIGS[variant]["block"]
    layers = RESNET_CONFIGS[variant]["layers"]

    model = ResNet3D(
        block=block,
        layers=layers,
        in_channels=in_channels,
        n_classes=n_classes,
        base_channels=base_channels,
    )
    return model
