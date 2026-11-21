"""Classic U-Net (Ronneberger et al., MICCAI 2015), trained from scratch."""
import torch
import torch.nn as nn

from ..losses import bce_dice
from .common import SegModel


def double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(SegModel):
    def __init__(self, base=64):
        super().__init__(imagenet_norm=True)
        ch = [base * 2 ** i for i in range(5)]
        self.downs = nn.ModuleList([double_conv(3, ch[0])] + [double_conv(ch[i], ch[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([double_conv(2 * ch[i], ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        x = self.normalize(x)
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x if i == 0 else self.pool(x))
            skips.append(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips[:-1])):
            x = dec(torch.cat([up(x), skip], 1))
        return self.head(x)

    def compute_loss(self, x, y):
        loss = bce_dice(self(x), y)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict(self, x):
        return torch.sigmoid(self(x))
