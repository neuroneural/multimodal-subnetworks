import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
from torch.utils.checkpoint import checkpoint_sequential
import json
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def set_channel_num(config, in_channels, n_classes, channels):
    """
    Takes a configuration json for a convolutional neural network of MeshNet architecture and changes it to have the specified number of input channels, output classes, and number of channels that each layer except the input and output layers have.

    Args:
        config (dict): The configuration json for the network.
        in_channels (int): The number of input channels.
        n_classes (int): The number of output classes.
        channels (int): The number of channels that each layer except the input and output layers will have.

    Returns:
        dict: The updated configuration json.
    """
    # input layer
    config["layers"][0]["in_channels"] = in_channels
    config["layers"][0]["out_channels"] = channels

    # output layer
    config["layers"][-1]["in_channels"] = channels
    config["layers"][-1]["out_channels"] = n_classes

    # hidden layers
    for layer in config["layers"][1:-1]:
        layer["in_channels"] = layer["out_channels"] = channels

    return config


def construct_layer(dropout_p=0, bnorm=True, gelu=False, *args, **kwargs):
    """Constructs a configurable Convolutional block with Batch Normalization and Dropout.

    Args:
    dropout_p (float): Dropout probability. Default is 0.
    bnorm (bool): Whether to include batch normalization. Default is True.
    gelu (bool): Whether to use GELU activation. Default is False.
    *args: Additional positional arguments to pass to nn.Conv3d.
    **kwargs: Additional keyword arguments to pass to nn.Conv3d.

    Returns:
    nn.Sequential: A sequential container of Convolutional block with optional Batch Normalization and Dropout.
    """
    layers = []
    layers.append(nn.Conv3d(*args, **kwargs))
    if bnorm:
        # track_running_stats=False is needed to run the forward mode AD
        layers.append(
            nn.BatchNorm3d(kwargs["out_channels"], track_running_stats=True)
        )
    layers.append(nn.ELU(inplace=True) if gelu else nn.ReLU(inplace=True))
    if dropout_p > 0:
        layers.append(nn.Dropout3d(dropout_p))
    return nn.Sequential(*layers)


def init_weights(model):
    """Set weights to be xavier normal for all Convs"""
    for m in model.modules():
        if isinstance(
            m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu"
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

class BasicBlock3D(nn.Module):
    """3D ResNet basic block with memory optimizations"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels, track_running_stats=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels, track_running_stats=True)
        self.dropout = nn.Dropout3d(dropout_p)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += residual
        return F.relu(out)

class ResNet3D(nn.Module):
    """3D ResNet with same interface as original MeshNet"""
    def __init__(self, in_channels, n_classes, channels, config_file=None):
        super().__init__()
        # Configurable parameters (maintaining compatibility)
        self.in_channels = in_channels
        self.n_classes = 1  # Binary classification
        self.channels = channels
        
        # Initial layers
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(channels, channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(channels, channels*2, blocks=2, stride=2)
        self.layer3 = self._make_layer(channels*2, channels*4, blocks=2, stride=2)
        self.layer4 = self._make_layer(channels*4, channels*8, blocks=2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(channels*8, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock3D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))
        return x

class enMesh_checkpoint(ResNet3D):
    """Memory-efficient version with gradient checkpointing"""
    def train_forward(self, x):
        # Forward pass with checkpointing
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = checkpoint_sequential(self.layer1, len(self.layer1), x)
        x = checkpoint_sequential(self.layer2, len(self.layer2), x)
        x = checkpoint_sequential(self.layer3, len(self.layer3), x)
        x = checkpoint_sequential(self.layer4, len(self.layer4), x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc(x))
        return x
    
    def eval_forward(self, x):
        with torch.inference_mode():
            return super().forward(x)
    
    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)

class enMesh(enMesh_checkpoint):
    """Most memory-efficient version with manual gradient management"""
    def __init__(
        self,
        in_channels,
        n_classes,
        channels,
        config_file=None,
        optimize_inline=False,
    ):
        super().__init__(in_channels, n_classes, channels, config_file)
        self.n_classes = 1  # Binary classification
        self.optimize_inline = optimize_inline
        if optimize_inline:
            self.optimizers = [
                torch.optim.Adam(self.conv1.parameters(), lr=0.02),
                torch.optim.Adam(self.layer1.parameters(), lr=0.02),
                torch.optim.Adam(self.layer2.parameters(), lr=0.02),
                torch.optim.Adam(self.layer3.parameters(), lr=0.02),
                torch.optim.Adam(self.layer4.parameters(), lr=0.02),
                torch.optim.Adam(self.fc.parameters(), lr=0.02)
            ]

    def get_grads(self, grads):
        def show(self, grad_input, grad_output):
            grads["in"] = grad_input
            grads["out"] = grad_output

        return show

    def set_requires_grad_layer(self, layer, flag, trainBN=True):
        layer.train(flag)
        for x in layer.parameters():
            if not flag:
                del x.grad
                x.detach()
            x.grad = [None, x.grad][flag]
            x.requires_grad = flag
        if (
            trainBN
            and isinstance(layer, torch.nn.Sequential)
            and isinstance(layer[1], torch.nn.BatchNorm3d)
        ):
            layer[1].training = True
            layer[1].requires_grad = True

    def unset_grad(self, layer):
        self.set_requires_grad_layer(layer, False)

    def set_grad(self, layer):
        self.set_requires_grad_layer(layer, True)

    def dump_tensors(gpu_only=True):
        # torch.cuda.empty_cache()
        total_size = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if not gpu_only or obj.is_cuda:
                        del obj
                        gc.collect()
                elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                    if not gpu_only or obj.is_cuda:
                        del obj
                        gc.collect()
            except Exception as e:
                pass

    def eval_forward(self, x):
        """Forward pass"""
        with torch.inference_mode():
            return super().forward(x)

    def forward(self, x, y=None, loss=None, verbose=False):
        if self.training:
            return self.backforward(x, y, loss, verbose=verbose)
        else:
            return self.eval_forward(x)

    def backforward(self, x, y, loss, verbose=False):
        if verbose:
            h = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"total    : {info.total}")
            print(f"free     : {info.free}")
            print(f"used     : {info.used}")
            print(f"used fr  : {info.used/info.total}")

        gradients = {}
        layers = [self.conv1, self.bn1, self.maxpool,
                 *self.layer1, *self.layer2, 
                 *self.layer3, *self.layer4,
                 self.avgpool, self.fc]
        
        for p in layers:
            self.unset_grad(p)

        grads = {}
        handle = layers[-1].register_full_backward_hook(self.get_grads(grads))

        self.set_grad(layers[-1])
        input = x
        input.requires_grad = False
        
        # Forward pass
        input = self.train_forward(input)
        y_hat = input
        input.requires_grad_()
        input.detach()

        if verbose:
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"used fr  : {info.used/info.total}")

        # Binary classification loss
        output = F.binary_cross_entropy(input, y.float())
        output.backward()
        output.detach()
        lss_value = output
        del output
        del input
        self.unset_grad(layers[-1])
        handle.remove()

        dloss_dx2 = grads["out"][0]
        del grads["in"]

        if verbose:
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"used fr  : {info.used/info.total}")
            print("*" * 20)

        # Backward pass through each layer
        for i in range(len(layers) - 1, -1, -1):
            input = x.detach().clone()
            input.requires_grad = False
            grads = {}
            handle = layers[i].register_full_backward_hook(
                self.get_grads(grads)
            )
            self.set_grad(layers[i])

            # Recompute forward pass up to current layer
            for j in range(0, i + 1):
                if j == i:
                    input.detach()
                    input.requires_grad_()
                input = layers[j](input)

            input.detach()
            torch.autograd.backward(input, dloss_dx2)

            del dloss_dx2
            dloss_dx2 = grads["in"][0]

            if self.optimize_inline:
                self.optimizers[i].step()
                self.optimizers[i].zero_grad(set_to_none=True)
            else:
                gradients[i] = [x.grad for x in layers[i].parameters()]

            self.unset_grad(layers[i])
            handle.remove()
            del input.grad
            del x.grad
            del input
            x.requires_grad = False
            
        del dloss_dx2
        self.eval()
        
        if not self.optimize_inline:
            for i in range(len(layers)):
                for p, g in zip(layers[i].parameters(), gradients[i]):
                    p.grad = g
        
        del layers
        if verbose:
            info = nvmlDeviceGetMemoryInfo(h)
            print(f"{i} used fr  : {info.used/info.total}")

        return lss_value, y_hat

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = 64  # Increased from original 5 for ResNet
    cubesize = 256
    classes = 1    # Binary classification
    batch = 1

    # Note: config_file is now optional since we're using ResNet architecture
    emodel = enMesh_checkpoint(1, classes, channels, config_file=None).to(device)
