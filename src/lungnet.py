import torch
import torch.nn as nn

class LungNet(nn.Module):
    def __init__(self, num_classes, res3_blocks, res4_blocks, res5_blocks):
        super(LungNet, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)

        self.res3 = self._make_layer(128, 256, num_blocks=res3_blocks)

        self.res4 = self._make_layer(256, 512, num_blocks=res4_blocks)

        self.res5 = self._make_layer(512, 1024, num_blocks=res5_blocks)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(self._make_res_block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
