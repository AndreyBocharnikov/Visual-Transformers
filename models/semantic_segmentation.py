import typing as tp
import copy

from utils import change_names

import torch
import torch.nn as nn

from models.visual_transformer import VisualTransformer, FilterBasedTokenizer, RecurrentTokenizer, Transformer, \
    Projector
import torchvision.models as models


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

    def __init__(self, n_classes, u2_cs=256, u3_cs=256, u4_cs=256, u5_cs=256):
        super().__init__()
        self.u2 = nn.Conv2d(u2_cs, 128, kernel_size=3, padding=1)
        self.u3 = SemanticSegmentationBranch.upsampling_stage(u3_cs, 128)
        self.u4 = make_layer(SemanticSegmentationBranch.upsampling_stage(u4_cs, 128),
                             SemanticSegmentationBranch.upsampling_stage(128, 128), 1)
        self.u5 = make_layer(SemanticSegmentationBranch.upsampling_stage(u5_cs, 128),
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


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def load_resnet(backbone):
      pretrained_weights = models.resnet50(pretrained=True).state_dict()
      my_state_dict = change_names(pretrained_weights)
      backbone.load_state_dict(my_state_dict)
      backbone.apply(set_bn_eval)


class PanopticFPN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = ResNet50Backbone()
        load_resnet(self.backbone)

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
    def __init__(self, n_classes, n_visual_tokens=8):
        super().__init__()
        self.n_visual_tokens = n_visual_tokens
        self.backbone = ResNet50Backbone()
        load_resnet(self.backbone)

        self.tokenizer2 = FilterBasedTokenizer(256, 1024, n_visual_tokens)
        self.tokenizer3 = FilterBasedTokenizer(512, 1024, n_visual_tokens)
        self.tokenizer4 = FilterBasedTokenizer(1024, 1024, n_visual_tokens)
        self.tokenizer5 = FilterBasedTokenizer(2048, 1024, n_visual_tokens)
        self.transformer = Transformer(1024)
        self.projector2 = Projector(256, 1024)
        self.projector3 = Projector(512, 1024)
        self.projector4 = Projector(1024, 1024)
        self.projector5 = Projector(2048, 1024)
        self.upsample = nn.Upsample(scale_factor=2)
        self.ss_branch = SemanticSegmentationBranch(n_classes, 256, 512, 1024, 2048)

    def forward(self, X):
        bs, ch, h, w = X.shape
        c2, c3, c4, c5 = self.backbone(X)
        c2, c3, c4, c5 = torch.flatten(c2, start_dim=2), torch.flatten(c3, start_dim=2), torch.flatten(c4, start_dim=2), torch.flatten(c5, start_dim=2)
        visual_tokens2 = self.tokenizer2(c2)
        visual_tokens3 = self.tokenizer3(c3)
        visual_tokens4 = self.tokenizer4(c4)
        visual_tokens5 = self.tokenizer5(c5)
        all_visual_tokens = torch.cat((visual_tokens2, visual_tokens3, visual_tokens4, visual_tokens5), dim=2)
        t2, t3, t4, t5 = torch.split(self.transformer(all_visual_tokens), self.n_visual_tokens, dim=2)
        p5 = self.projector5(c5, t5).view(bs, -1, h // 32, w // 32)
        p4 = self.projector4(c4, t4).view(bs, -1, h // 16, w // 16)
        p3 = self.projector3(c3, t3).view(bs, -1, h // 8, w // 8)
        p2 = self.projector2(c2, t2).view(bs, -1, h // 4, w // 4)
        return self.ss_branch(p2, p3, p4, p5)
