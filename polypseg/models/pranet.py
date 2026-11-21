"""PraNet: Parallel Reverse Attention Network (Fan et al., MICCAI 2020).

Port of the official lib/PraNet_Res2Net.py (https://github.com/DengPingFan/PraNet).
The architecture is unchanged; the Res2Net-50 (26w_4s) ImageNet backbone is
loaded through timm because the original weight URL is no longer maintained.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import structure_loss
from .common import SegModel


class BasicConv2d(nn.Module):
    # As in the official code, no ReLU is applied inside this block.
    def __init__(self, cin, cout, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class RFB_modified(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.branch0 = BasicConv2d(cin, cout, 1)

        def branch(k):
            return nn.Sequential(
                BasicConv2d(cin, cout, 1),
                BasicConv2d(cout, cout, (1, k), padding=(0, k // 2)),
                BasicConv2d(cout, cout, (k, 1), padding=(k // 2, 0)),
                BasicConv2d(cout, cout, 3, padding=k, dilation=k),
            )

        self.branch1, self.branch2, self.branch3 = branch(3), branch(5), branch(7)
        self.conv_cat = BasicConv2d(4 * cout, cout, 3, padding=1)
        self.conv_res = BasicConv2d(cin, cout, 1)

    def forward(self, x):
        x_cat = self.conv_cat(torch.cat([self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)], 1))
        return F.relu(x_cat + self.conv_res(x))


class Aggregation(nn.Module):
    """Parallel partial decoder."""

    def __init__(self, c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(c, c, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(c, c, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(c, c, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(c, c, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * c, 2 * c, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * c, 2 * c, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * c, 3 * c, 3, padding=1)
        self.conv4 = BasicConv2d(3 * c, 3 * c, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * c, 1, 1)

    def forward(self, x1, x2, x3):
        up = self.up
        x2_1 = self.conv_upsample1(up(x1)) * x2
        x3_1 = self.conv_upsample2(up(up(x1))) * self.conv_upsample3(up(x2)) * x3
        x2_2 = self.conv_concat2(torch.cat([x2_1, self.conv_upsample4(up(x1))], 1))
        x3_2 = self.conv_concat3(torch.cat([x3_1, self.conv_upsample5(up(x2_2))], 1))
        return self.conv5(self.conv4(x3_2))


def _interp(x, s):
    return F.interpolate(x, scale_factor=s, mode="bilinear")


class PraNet(SegModel):
    def __init__(self, channel=32, pretrained=True):
        super().__init__(imagenet_norm=True)
        self.backbone = timm.create_model("res2net50_26w_4s", pretrained=pretrained,
                                          features_only=True, out_indices=(1, 2, 3, 4))
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        self.agg1 = Aggregation(channel)
        self.ra4_conv1 = BasicConv2d(2048, 256, 1)
        self.ra4_conv2 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, 1)
        self.ra3_conv1 = BasicConv2d(1024, 64, 1)
        self.ra3_conv2 = BasicConv2d(64, 64, 3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, 3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, 3, padding=1)
        self.ra2_conv1 = BasicConv2d(512, 64, 1)
        self.ra2_conv2 = BasicConv2d(64, 64, 3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, 3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, 3, padding=1)

    def forward(self, x):
        _, x2, x3, x4 = self.backbone(self.normalize(x))
        ra5_feat = self.agg1(self.rfb4_1(x4), self.rfb3_1(x3), self.rfb2_1(x2))
        lateral_map_5 = _interp(ra5_feat, 8)

        crop_4 = _interp(ra5_feat, 0.25)
        x = (1 - torch.sigmoid(crop_4)).expand(-1, 2048, -1, -1) * x4
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        x = self.ra4_conv5(x) + crop_4
        lateral_map_4 = _interp(x, 32)

        crop_3 = _interp(x, 2)
        x = (1 - torch.sigmoid(crop_3)).expand(-1, 1024, -1, -1) * x3
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        x = self.ra3_conv4(x) + crop_3
        lateral_map_3 = _interp(x, 16)

        crop_2 = _interp(x, 2)
        x = (1 - torch.sigmoid(crop_2)).expand(-1, 512, -1, -1) * x2
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        x = self.ra2_conv4(x) + crop_2
        lateral_map_2 = _interp(x, 8)
        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2

    def compute_loss(self, x, y):
        maps = self(x)
        losses = [structure_loss(m, y) for m in maps]
        loss = sum(losses)
        return loss, {"loss": loss.item(), "loss_2": losses[-1].item()}

    @torch.no_grad()
    def predict(self, x):
        return torch.sigmoid(self(x)[-1])
