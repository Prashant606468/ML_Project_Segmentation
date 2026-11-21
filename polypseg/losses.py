import torch
import torch.nn.functional as F


def bce_dice(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    p = torch.sigmoid(logits)
    inter = (p * mask).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + mask.sum((2, 3)) + 1)
    return bce + dice.mean()


def structure_loss(logits, mask):
    """Boundary-weighted BCE + IoU loss from PraNet (Fan et al., MICCAI 2020).

    Pixels near the mask boundary (where the local average differs from the
    pixel value) get up to 6x weight.
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    wbce = (weit * wbce).sum((2, 3)) / weit.sum((2, 3))
    p = torch.sigmoid(logits)
    inter = (p * mask * weit).sum((2, 3))
    union = ((p + mask) * weit).sum((2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def edge_map(mask, width=3):
    """Morphological gradient of a binary mask -> thin boundary band."""
    k = 2 * (width // 2) + 1
    dil = F.max_pool2d(mask, k, stride=1, padding=k // 2)
    ero = -F.max_pool2d(-mask, k, stride=1, padding=k // 2)
    return (dil - ero).clamp(0, 1)


def edge_loss(edge_logits, mask):
    e = edge_map(mask)
    # Edges are a small fraction of pixels -> positive weighting + dice.
    pos = e.mean().clamp(min=1e-3)
    bce = F.binary_cross_entropy_with_logits(edge_logits, e, pos_weight=(1 - pos) / pos)
    p = torch.sigmoid(edge_logits)
    dice = 1 - (2 * (p * e).sum((2, 3)) + 1) / (p.sum((2, 3)) + e.sum((2, 3)) + 1)
    return bce + dice.mean()
