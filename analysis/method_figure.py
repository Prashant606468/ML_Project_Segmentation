"""Architecture diagram of YOLO-Diff for the report -> report/figures/method.png"""
import os
import sys

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from style import MUTED, TEXT, setup  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e4e3df"


def box(ax, x, y, w, h, text, color, fs=8.5, alpha=0.18):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=TEXT)


def arrow(ax, p, q, color=MUTED, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, linewidth=1.2,
                                 linestyle=ls, shrinkA=2, shrinkB=2))


def main():
    plt = setup()
    fig, ax = plt.subplots(figsize=(10, 4.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.55, 3.6)
    ax.axis("off")

    box(ax, 0.1, 2.3, 1.1, 0.8, "Image\n352×352", GRAY, alpha=0.6)
    box(ax, 1.6, 2.3, 1.7, 0.8, "YOLOv8s backbone\n(layers 0–9, 5.1M)", BLUE)
    box(ax, 3.7, 2.3, 1.5, 0.8, "FPN neck\nC$_2$…C$_{32}$ (64 ch)", BLUE)
    arrow(ax, (1.2, 2.7), (1.6, 2.7))
    arrow(ax, (3.3, 2.7), (3.7, 2.7))
    ax.text(4.45, 3.3, "computed once per image", ha="center", fontsize=8, color=MUTED, style="italic")

    box(ax, 0.05, 0.5, 1.3, 0.8, "$x_t$ noisy mask\n(0 in YOLO-Seg)", GRAY, fs=8, alpha=0.6)
    box(ax, 0.05, 1.45, 1.3, 0.5, "timestep $t$", GRAY, fs=8, alpha=0.6)
    box(ax, 1.6, 0.1, 4.6, 1.85, "", ORANGE, alpha=0.10)
    ax.text(3.9, 0.3, "Conditional denoising U-Net (9.2M)", ha="center", fontsize=9, color=TEXT, weight="bold")
    xs = [1.8, 2.55, 3.3, 4.05, 4.8, 5.55]
    labels = ["patchify\n2×2", "s2", "s4", "s8", "s16", "s32\n+attn"]
    for x, l in zip(xs, labels):
        box(ax, x, 0.6, 0.55, 0.8, l, ORANGE, fs=7.5, alpha=0.25)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 0.55, 1.0), (b, 1.0))
    arrow(ax, (1.35, 0.9), (1.8, 0.9))
    arrow(ax, (1.35, 1.7), (1.8, 1.45), ls="--")
    for x in xs[1:]:
        arrow(ax, (4.45, 2.3), (x + 0.27, 1.4), color=BLUE, ls=":")
    ax.text(6.3, 2.05, "features added at every scale\n(encoder + decoder); pooled C$_{32}$\nand $t$ modulate blocks via FiLM",
            fontsize=7.5, color=MUTED, va="center")

    box(ax, 6.6, 0.6, 1.3, 0.8, "decoder +\nPixelShuffle", ORANGE, fs=8)
    arrow(ax, (6.2, 1.0), (6.6, 1.0))
    box(ax, 8.3, 1.0, 1.55, 0.55, "mask logits $\\hat{x}_0$", AQUA, fs=8)
    box(ax, 8.3, 0.2, 1.55, 0.55, "edge map (train only)", AQUA, fs=8)
    arrow(ax, (7.9, 1.1), (8.3, 1.27))
    arrow(ax, (7.9, 0.9), (8.3, 0.47))
    # DDIM loop: predicted mask -> next noisy mask, routed below the network
    ax.plot([9.85, 9.97, 9.97, 0.7], [1.27, 1.27, -0.2, -0.2], color=MUTED, lw=1.1, ls="--")
    arrow(ax, (0.7, -0.2), (0.7, 0.5), ls="--")
    ax.text(5.0, -0.42, "DDIM update $x_{t'} = \\sqrt{\\bar\\alpha_{t'}}\\,\\hat{x}_0 + \\sqrt{1-\\bar\\alpha_{t'}}\\,\\hat\\epsilon$, "
            "repeated for S steps (diffusion only)", ha="center", fontsize=8, color=MUTED)
    os.makedirs("report/figures", exist_ok=True)
    fig.savefig("report/figures/method.png", dpi=250, bbox_inches="tight")


if __name__ == "__main__":
    main()
