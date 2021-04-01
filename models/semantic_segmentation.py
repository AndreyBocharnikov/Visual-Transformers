import torch
import torch.nn as nn

from models.visual_transformer import VisualTransformer, FilterBasedTokenizer, RecurrentTokenizer


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self):
        pass


class SemanticSegmentationBranch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class PanopticFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18Backbone()
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
