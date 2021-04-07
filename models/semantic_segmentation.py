import typing as tp
import copy

import torch
import torch.nn as nn

from models.visual_transformer import VisualTransformer, FilterBasedTokenizer, RecurrentTokenizer


def make_layer(first_layer, type_of_rest_layers, n_rest_layers):
    layers = [first_layer]
    layers += [copy.deepcopy(type_of_rest_layers)] * n_rest_layers
    return nn.Sequential(*layers)


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

        self.conv2 = make_layer(Bottleneck(64, 64, 256, downsample=True), Bottleneck(256, 64, 256), 2)
        #    nn.Sequential(Bottleneck(64, 64, 256, downsample=True),
        #                           Bottleneck(256, 64, 256),
        #                           Bottleneck(256, 64, 256))

        self.conv3 = make_layer(Bottleneck(256, 128, 512, stride=2), Bottleneck(512, 128, 512), 3)
            #nn.Sequential(Bottleneck(256, 128, 512, stride=2),
            #                       Bottleneck(512, 128, 512),
            #                       Bottleneck(512, 128, 512),
            #                       Bottleneck(512, 128, 512))

        self.conv4 = make_layer(Bottleneck(512, 256, 1024, stride=2), Bottleneck(1024, 256, 1024), 5)
            #nn.Sequential(Bottleneck(512, 256, 1024, stride=2),
            #                       Bottleneck(1024, 256, 1024),
            #                       Bottleneck(1024, 256, 1024),
            #                       Bottleneck(1024, 256, 1024),
            #                       Bottleneck(1024, 256, 1024),
            #                       Bottleneck(1024, 256, 1024))

        self.conv5 = make_layer(Bottleneck(1024, 512, 2048, stride=2), Bottleneck(2048, 512, 2048), 2)
            #nn.Sequential(Bottleneck(1024, 512, 2048, stride=2),
            #                       Bottleneck(2048, 512, 2048),
            #                       Bottleneck(2048, 512, 2048))

    def forward(self, X):
        c1 = self.conv1(X)  # bs, 64, h/4, w/4
        c2 = self.conv2(c1)  # bs, 256, h/4, w/4
        c3 = self.conv3(c2)  # bs, 512, h/8, w/8
        c4 = self.conv4(c3)  # bs, 1024, h/8, w/8
        c5 = self.conv5(c4)  # bs, 2048, h/16, w/16
        return c2, c3, c4, c5


class SemanticSegmentationBranch(nn.Module):
    @staticmethod
    def upsampling_stage(in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                             nn.GroupNorm(32, out_channels),
                             nn.ReLU(),
                             nn.Upsample(scale_factor=2, mode="bilinear"))

    def __init__(self, n_classes):
        super().__init__()
        self.u2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.u3 = SemanticSegmentationBranch.upsampling_stage(256, 128)
        self.u4 = make_layer(SemanticSegmentationBranch.upsampling_stage(256, 128),
                             SemanticSegmentationBranch.upsampling_stage(128, 128), 1)
        self.u5 = make_layer(SemanticSegmentationBranch.upsampling_stage(256, 128),
                             SemanticSegmentationBranch.upsampling_stage(128, 128), 2)

        self.final_conv = nn.Conv2d(128, n_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear")
        self.softmax = nn.Softmax(dim=1)

    def forward(self, p2, p3, p4, p5):
        g2 = self.u2(p2)
        g3 = self.u3(p3)
        g4 = self.u4(p4)
        g5 = self.u5(p5)

        result = self.final_conv(g2 + g3 + g4 + g5)
        result = self.upsample(result)
        return self.softmax(result)


class PanopticFPN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = ResNet50Backbone()
        # TODO pretrain + eval mode
        self.skip_con_conv2 = nn.Conv2d(256, 256, kernel_size=1)
        self.skip_con_conv3 = nn.Conv2d(512, 256, kernel_size=1)
        self.skip_con_conv4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.skip_con_conv5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.ss_branch = SemanticSegmentationBranch(n_classes)

    def forward(self, X):
        c2, c3, c4, c5 = self.backbone(X)
        p5 = self.skip_con_conv5(c5)
        p4 = self.skip_con_conv4(c4) + self.upsample(p5)
        p3 = self.skip_con_conv3(c3) + self.upsample(p4)
        p2 = self.skip_con_conv2(c2) + self.upsample(p3)
        return self.ss_branch(p2, p3, p4, p5)


class VT_FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.vt2 = VisualTransformer(FilterBasedTokenizer(256, 1024, 8), use_projector=True)
        self.vt3 = VisualTransformer(FilterBasedTokenizer(512, 1024, 8), use_projector=True)
        self.vt4 = VisualTransformer(FilterBasedTokenizer(1024, 1024, 8), use_projector=True)
        self.vt5 = VisualTransformer(FilterBasedTokenizer(2048, 1024, 8), use_projector=True)


    def forward(self, X):
        c2, c3, c4, c5 = self.backbone(X)
        visual_tokens2 = self.vt2(c2)
        visual_tokens3 = self.vt3(c3)
        visual_tokens4 = self.vt4(c4)
        visual_tokens5 = self.vt5(c5)

