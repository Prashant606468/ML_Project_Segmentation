import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegModel(nn.Module):
    """Common interface: inputs are RGB in [0, 1].

    compute_loss(x, y) -> (loss, log dict); predict(x) -> foreground probability (B,1,H,W).
    `backbone` (if any) is trained with a reduced learning rate.
    """

    def __init__(self, imagenet_norm):
        super().__init__()
        self.backbone = None  # subclasses may replace with an nn.Module
        self.imagenet_norm = imagenet_norm
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def normalize(self, x):
        return (x - self.mean) / self.std if self.imagenet_norm else x
