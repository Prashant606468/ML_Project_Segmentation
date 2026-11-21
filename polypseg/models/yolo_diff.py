"""YOLO-backbone-guided conditional diffusion for polyp segmentation (proposed).

Pipeline
    image --YOLOv8 backbone (layers 0-9)--> multi-scale features (strides 2..32)
          --FPN-lite neck--> conditioning maps C2..C32 (+ global vector)
    x_t (noisy mask) + t --conditional denoising U-Net (features injected at every scale)--> x0 logits

The YOLO features are computed once per image and reused at every denoising
step, so iterative sampling only re-runs the small denoiser.

The denoiser predicts the clean mask x0 (as logits) instead of the noise, so it
can be trained with segmentation losses; sampling uses deterministic DDIM.

Ablation switches
    diffusion=False : same network with x_t = 0 and t = 0, i.e. a one-shot
                      segmentation decoder ("yolo_seg").
    boundary=True   : boundary guidance = PraNet-style boundary-weighted loss
                      + auxiliary boundary head supervised with mask edges.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import bce_dice, edge_loss, structure_loss
from .common import SegModel


class YoloBackbone(nn.Module):
    """Layers 0-9 (backbone up to SPPF) of an Ultralytics YOLOv8 detector.

    YOLO was pretrained on RGB inputs in [0, 1], so no mean/std normalization.
    """

    OUT_IDX = (0, 2, 4, 6, 9)  # strides 2, 4, 8, 16, 32

    def __init__(self, weights):
        super().__init__()
        from ultralytics import YOLO

        self.layers = nn.ModuleList(list(YOLO(weights).model.model[:10])).float()
        for p in self.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            self.channels = [f.shape[1] for f in self(torch.zeros(1, 3, 64, 64))]

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.OUT_IDX:
                feats.append(x)
        return feats


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class Neck(nn.Module):
    """Top-down FPN that maps every backbone level to `c` channels."""

    def __init__(self, in_chs, c):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(ci, c, 1) for ci in in_chs])
        self.out = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 3, padding=1), gn(c), nn.SiLU()) for _ in in_chs])

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lat, feats)]
        for i in range(len(lat) - 2, -1, -1):
            lat[i] = lat[i] + F.interpolate(lat[i + 1], size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
        return [o(l) for o, l in zip(self.out, lat)]


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], 1)


class ResBlock(nn.Module):
    """GN-SiLU-Conv residual block with FiLM (scale/shift) from the embedding."""

    def __init__(self, cin, cout, emb_dim):
        super().__init__()
        self.norm1, self.conv1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.norm2, self.conv2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, 1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = gn(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.norm(x).flatten(2).transpose(1, 2)
        s, _ = self.attn(s, s, s, need_weights=False)
        return x + s.transpose(1, 2).reshape(b, c, h, w)


class Denoiser(nn.Module):
    """Conditional U-Net over the mask, operating at strides 2..32.

    The full-resolution mask is patchified (2x2) at the input and restored by
    PixelShuffle at the output. Conditioning maps are added at every encoder
    and decoder level.
    """

    def __init__(self, cond_c, chs=(32, 64, 128, 192, 256), emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.g_mlp = nn.Sequential(nn.Linear(cond_c, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.stem = nn.Conv2d(1, chs[0], 2, stride=2)
        self.enc_cond = nn.ModuleList([nn.Conv2d(cond_c, c, 1) for c in chs])
        self.enc = nn.ModuleList([ResBlock(c, c, emb_dim) for c in chs])
        self.down = nn.ModuleList([nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1) for i in range(len(chs) - 1)])
        self.mid1 = ResBlock(chs[-1], chs[-1], emb_dim)
        self.mid_attn = SelfAttention(chs[-1])
        self.mid2 = ResBlock(chs[-1], chs[-1], emb_dim)
        self.up = nn.ModuleList([nn.Conv2d(chs[i + 1], chs[i], 3, padding=1) for i in range(len(chs) - 1)])
        self.dec_cond = nn.ModuleList([nn.Conv2d(cond_c, c, 1) for c in chs[:-1]])
        self.dec = nn.ModuleList([ResBlock(2 * c, c, emb_dim) for c in chs[:-1]])
        self.out_norm = gn(chs[0])
        self.mask_head = nn.Sequential(nn.Conv2d(chs[0], 4, 3, padding=1), nn.PixelShuffle(2))
        self.edge_head = nn.Sequential(nn.Conv2d(chs[0], 4, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, x_t, t, conds, g):
        emb = self.t_mlp(timestep_embedding(t, self.emb_dim)) + self.g_mlp(g)
        h = self.stem(x_t)
        skips = []
        for i, (cproj, block) in enumerate(zip(self.enc_cond, self.enc)):
            h = block(h + cproj(conds[i]), emb)
            if i < len(self.down):
                skips.append(h)
                h = self.down[i](h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for i in reversed(range(len(self.dec))):
            h = self.up[i](F.interpolate(h, size=skips[i].shape[-2:], mode="nearest"))
            h = self.dec[i](torch.cat([h + self.dec_cond[i](conds[i]), skips[i]], 1), emb)
        h = F.silu(self.out_norm(h))
        return self.mask_head(h), self.edge_head(h)


def cosine_alphas_cumprod(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


class YoloDiff(SegModel):
    def __init__(self, weights="weights/yolov8s.pt", diffusion=True, boundary=True, cond_c=64,
                 T=1000, sample_steps=10, n_samples=1, scale=1.0):
        super().__init__(imagenet_norm=False)
        self.backbone = YoloBackbone(weights)
        self.neck = Neck(self.backbone.channels, cond_c)
        self.denoiser = Denoiser(cond_c)
        self.diffusion, self.boundary = diffusion, boundary
        self.T, self.sample_steps, self.n_samples, self.scale = T, sample_steps, n_samples, scale
        self.register_buffer("ab", cosine_alphas_cumprod(T), persistent=False)

    def encode(self, x):
        conds = self.neck(self.backbone(x))
        return conds, conds[-1].mean((2, 3))

    def denoise(self, x_t, t, conds, g):
        return self.denoiser(x_t, t, conds, g)

    def compute_loss(self, x, y):
        conds, g = self.encode(x)
        b = x.shape[0]
        if self.diffusion:
            t = torch.randint(0, self.T, (b,), device=x.device)
            ab = self.ab[t].view(b, 1, 1, 1)
            x0 = (2 * y - 1) * self.scale
            x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)
        else:
            t = torch.zeros(b, dtype=torch.long, device=x.device)
            x_t = torch.zeros_like(y)
        logits, edge_logits = self.denoise(x_t, t, conds, g)
        logs = {}
        if self.boundary:
            seg = structure_loss(logits, y)
            edge = edge_loss(edge_logits, y)
            loss = seg + 0.5 * edge
            logs["edge"] = edge.item()
        else:
            loss = seg = bce_dice(logits, y)
        logs.update(loss=loss.item(), seg=seg.item())
        return loss, logs

    @torch.no_grad()
    def predict(self, x, steps=None, n_samples=None, generator=None):
        conds, g = self.encode(x)
        b = x.shape[0]
        if not self.diffusion:
            t = torch.zeros(b, dtype=torch.long, device=x.device)
            return torch.sigmoid(self.denoise(torch.zeros_like(x[:, :1]), t, conds, g)[0])
        steps = steps or self.sample_steps
        n = n_samples or self.n_samples
        if n > 1:  # ensemble of independent samples, run as one batch
            conds = [c.repeat(n, 1, 1, 1) for c in conds]
            g = g.repeat(n, 1)
        seq = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        shape = (b * n, 1, x.shape[2], x.shape[3])
        x_t = torch.randn(shape, device=x.device, generator=generator)
        for i, ti in enumerate(seq):
            t = torch.full((b * n,), ti, dtype=torch.long, device=x.device)
            logits = self.denoise(x_t, t, conds, g)[0]
            prob = torch.sigmoid(logits)
            if i == len(seq) - 1:
                break
            # DDIM (eta = 0) update using the predicted clean mask.
            x0 = (2 * prob - 1) * self.scale
            ab_t, ab_prev = self.ab[ti], self.ab[seq[i + 1]]
            eps = (x_t - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
            x_t = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return prob.view(n, b, *prob.shape[1:]).mean(0)
