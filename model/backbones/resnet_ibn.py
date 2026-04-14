import torch
import torch.nn as nn


class ResNet50IBNReID(nn.Module):
    def __init__(self, num_classes=1000, neck_feat='after', pretrained=True):
        super().__init__()

        base = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=pretrained)
        self.base = nn.Sequential(*list(base.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.neck_feat = neck_feat

    def forward(self, x, label=None, cam_label=None, view_label=None):
        feat_map = self.base(x)
        global_feat = self.gap(feat_map).view(feat_map.size(0), -1)
        feat = self.bnneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat