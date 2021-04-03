import torch
import torch.nn as nn

from models.visual_transformer import VisualTransformer, FilterBasedTokenizer, RecurrentTokenizer


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU()
        self.stride = stride

        self.plain_arch = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels))

        if stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_channels))

    def forward(self, X):
        if hasattr(self, "downsample"):
            identity = self.downsample(X)
        else:
            identity = X
        out = self.plain_arch(X)
        out += identity
        return self.relu(out)


def make_resnet14_backbone():
    return nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                         nn.BatchNorm2d(64),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                         BasicBlock(64, 64),
                         BasicBlock(64, 64),

                         BasicBlock(64, 128, stride=2),
                         BasicBlock(128, 128),

                         BasicBlock(128, 256, stride=2),
                         BasicBlock(256, 256))


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, n_classes, pooling):
        super().__init__()
        self.pooling = pooling
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, X):
        X = self.pooling(X)
        X = torch.flatten(X, start_dim=1)
        return self.fc(X)


class ResNet18(nn.Module):
    # ResNet18 aka Baseline
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = make_resnet14_backbone()
        self.conv_5 = nn.Sequential(BasicBlock(256, 512, stride=2),
                                    BasicBlock(512, 512))
        self.classification_head = ClassificationHead(in_dim=512, n_classes=n_classes, pooling=nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, X):
        X = self.backbone(X)
        X = self.conv_5(X)
        return self.classification_head(X)


class VT_ResNet18(nn.Module):
    def __init__(self, n_classes=1000):
        super().__init__()
        self.backbone = make_resnet14_backbone()
        tokenizer1 = FilterBasedTokenizer(feature_map_cs=256, visual_tokens_cs=1024, n_visual_tokens=8)
        self.vt1 = VisualTransformer(tokenizer1, use_projector=True)
        tokenizer2 = RecurrentTokenizer(feature_map_cs=256, visual_tokens_cs=1024)
        self.vt2 = VisualTransformer(tokenizer2, use_projector=False)

        self.classification_head = ClassificationHead(in_dim=8, n_classes=n_classes, pooling=nn.AdaptiveAvgPool1d(1))

    def forward(self, X):
        feature_map = self.backbone(X)
        feature_map = torch.flatten(feature_map, start_dim=2)
        feature_map, visual_tokens = self.vt1(feature_map, None)
        visual_tokens = self.vt2(feature_map, visual_tokens)

        return self.classification_head(visual_tokens)
