# NAS Paper does 3 augmentations, I am doing 21 so 7x more time
# NAS Paper does 10 epochs, I am doing 20 but each epoch sees half the data so same time
# NAS Paper takes 8 hours so my search should take 56 hours
# Will then select the best 9 architectures and train for more epochs using A-Softmax loss
# When evaluating performance, we predict maliganancy if n/2 of the models predict maliganancy
import torch
import torch.nn as nn
from net_sphere import AngleLinear

def conv3d_batchnorm_relu(in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
        nn.BatchNorm3d(out_channels),
        nn.ReLU()
    )

def conv3d_with_pooling(in_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
    padding = kernel_size // 2
    return conv3d_batchnorm_relu(in_channels, in_channels, kernel_size, stride, padding, dilation, groups, bias)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3d_batchnorm_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv3d_batchnorm_relu(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv3d_batchnorm_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.conv3(x)
        y = self.conv1(x) 
        y = self.conv2(y)
        y = y + z
        return y
    
class LungNet(nn.Module):
    def __init__(self, architecture):
        super(LungNet, self).__init__()
        self.conv1 = conv3d_batchnorm_relu(in_channels = 1, out_channels = 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv3d_batchnorm_relu(in_channels = 4, out_channels = 4, kernel_size=3, stride=1, padding=1)
        self.architecture = architecture
        self.last_channel = 4
        layers = []
        for stage in architecture:
            layers.append(conv3d_with_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                layers.append(ResidualBlock(self.last_channel, channel))
                self.last_channel = channel
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AvgPool3d(kernel_size=4, stride=4)
        self.fc = AngleLinear(in_features=self.last_channel, out_features=2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.layers(y)
        y = self.pool(y)
        y = y.view(y.size(0), -1)
        logits, aux = self.fc(y)
        return logits


    
