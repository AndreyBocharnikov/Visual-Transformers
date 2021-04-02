import typing as tp

import torch
import torch.nn as nn

from models.visual_transformer import VisualTransformer, FilterBasedTokenizer, RecurrentTokenizer


class Bottleneck(nn.Module):
    def __init__(self, in_channels, width, out_channels, stride=1, downsample=False):
        super().__init__()
        self.plain_arch = nn.Sequential(nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(width),
                                        nn.ReLU(),
                                        nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False),
                                        nn.BatchNorm2d(width),
                                        nn.ReLU(),
                                        nn.Conv2d(width, out_channels, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        if stride != 1 or downsample:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_channels))

    def forward(self, X):
        feature_map = self.plain_arch(X)
        if hasattr(self, "downsample"):
            identity = self.downsample(X)
        else:
            identity = X
        feature_map += identity
        return self.relu(feature_map)


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = nn.Sequential(Bottleneck(64, 64, 256, downsample=True),
                                   Bottleneck(256, 64, 256),
                                   Bottleneck(256, 64, 256))

        self.conv3 = nn.Sequential(Bottleneck(256, 128, 512, stride=2),
                                   Bottleneck(512, 128, 512),
                                   Bottleneck(512, 128, 512),
                                   Bottleneck(512, 128, 512))

        self.conv4 = nn.Sequential(Bottleneck(512, 256, 1024, stride=2),
                                   Bottleneck(1024, 256, 1024),
                                   Bottleneck(1024, 256, 1024),
                                   Bottleneck(1024, 256, 1024),
                                   Bottleneck(1024, 256, 1024),
                                   Bottleneck(1024, 256, 1024),)

        self.conv5 = nn.Sequential(Bottleneck(1024, 512, 2048, stride=2),
                                   Bottleneck(2048, 512, 2048),
                                   Bottleneck(2048, 512, 2048))

    def forward(self, X):
        c1 = self.conv1(X)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        return c2, c3, c4, c5


class SemanticSegmentationBranch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class PanopticFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.skip_con_conv2 = nn.Conv2d()
        self.skip_con_conv3 = nn.Conv2d()
        self.skip_con_conv4 = nn.Conv2d()
        self.skip_con_conv5 = nn.Conv2d()
        self.upsample = nn.Upsample(scale_factor=2)
        self.ss_branch = SemanticSegmentationBranch()

    def forward(self, X):
        c2, c3, c4, c5 = self.backbone(X)
        p5 = self.skip_con_conv5(c5)
        p4 = self.skip_con_conv4(c4) + self.upsample(c5)
        p3 = self.skip_con_conv3(c3) + self.upsample(c4)
        p2 = self.skip_con_conv2(c2) + self.upsample(c3)
        return



class VT_FPN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
