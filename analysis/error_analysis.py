"""Error analysis on the unseen datasets.

1. Dice by polyp size and by polyp/background contrast (seed-averaged per image).
2. Worst failure cases of the proposed model.
3. Qualitative figure: image, ground truth, U-Net, PraNet, YOLO-Diff and the
   diffusion model's per-pixel uncertainty (std over 8 samples), seed 0 models.

Requires results/dataset_stats.csv (analysis/dataset_stats.py) and eval CSVs.
"""
import os
import sys
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import load_run  # noqa: E402
from polypseg.data import IMG_SIZE, read_image, read_mask  # noqa: E402
from style import CONFIGS, config_of, setup  # noqa: E402

PROPOSED = "yolo_diff_v8s_bnd"
SIZE_BINS = [0, 0.02, 0.08, 1.0]
SIZE_LABELS = ["small (<2%)", "medium (2-8%)", "large (>8%)"]


def per_image(ds):
    """Seed-averaged per-image metrics for each configuration."""
    frames = []
    for f in glob(f"runs/*/eval/{ds}.csv"):
        d = pd.read_csv(f)
        d["config"] = config_of(f.split("/")[1])
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).groupby(["config", "image"], as_index=False).mean(numeric_only=True)


def bucket_tables(ds, stats):
    df = per_image(ds)
    if df.empty:
        return None
    st = stats[stats.dataset == f"{ds} (test)"][["image", "area_frac", "contrast_dE"]]
    df = df.merge(st, on="image")
    df["size"] = pd.cut(df.area_frac, SIZE_BINS, labels=SIZE_LABELS)
    df["contrast"] = pd.qcut(df.contrast_dE, 3, labels=["low", "medium", "high"])
    out = {}
    for key in ["size", "contrast"]:
        t = df.pivot_table(index="config", columns=key, values="dice", aggfunc="mean")
        t = t.reindex([c for c in CONFIGS if c in t.index])
        counts = df[df.config == df.config.iloc[0]][key].value_counts()
        t.columns = [f"{c} (n={counts[c]})" for c in t.columns]
        out[key] = t
    return df, out


def plot_size(tables, path):
    plt = setup()
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 4), sharey=True, squeeze=False)
    for ax, (ds, t) in zip(axes[0], tables.items()):
        n = len(t.index)
        w = 0.8 / n
        x = np.arange(len(t.columns))
        for i, c in enumerate(t.index):
            name, color = CONFIGS.get(c, (c, "#52514e"))
            ax.bar(x + (i - (n - 1) / 2) * w, t.loc[c].values, w * 0.9, color=color, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(t.columns)
        ax.set_title(f"{ds}: Dice by polyp size", loc="left")
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Mean Dice")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def overlay(img, mask, color):
    out = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, color, max(2, img.shape[0] // 150))
    return out


@torch.no_grad()
def qualitative(ds, images, path, device="cuda"):
    runs = {"U-Net": "runs/unet_s0", "PraNet": "runs/pranet_s0", "YOLO-Diff": f"runs/{PROPOSED}_s0"}
    runs = {k: v for k, v in runs.items() if os.path.exists(f"{v}/best.pt")}
    if f"runs/{PROPOSED}_s0" not in runs.values():
        return
    models = {k: load_run(v, device)[0] for k, v in runs.items()}
    plt = setup()
    cols = ["Image", "Ground truth"] + list(models) + ["YOLO-Diff uncertainty"]
    fig, axes = plt.subplots(len(images), len(cols), figsize=(2.3 * len(cols), 2.1 * len(images)), squeeze=False)
    for r, (name, note) in enumerate(images):
        img = read_image(f"data/TestDataset/{ds}/images/{name}")
        gt = read_mask(f"data/TestDataset/{ds}/masks/{name}")
        x = torch.from_numpy(cv2.resize(img, (IMG_SIZE, IMG_SIZE))).permute(2, 0, 1)[None].float().to(device) / 255
        panels = [img, overlay(img, gt, (40, 220, 40))]
        for k, m in models.items():
            prob = m.predict(x) if not getattr(m, "diffusion", False) else m.predict(
                x, generator=torch.Generator(device=device).manual_seed(0))
            prob = F.interpolate(prob, size=gt.shape, mode="bilinear")[0, 0].cpu().numpy()
            pred = prob > 0.5
            dice = 2 * (pred & gt.astype(bool)).sum() / max(pred.sum() + gt.sum(), 1)
            panels.append((overlay(overlay(img, gt, (40, 220, 40)), pred, (255, 60, 60)), f"Dice {dice:.2f}"))
        diff = models["YOLO-Diff"]
        gen = torch.Generator(device=device).manual_seed(1)
        samples = torch.cat([diff.predict(x, generator=gen) for _ in range(8)])
        unc = F.interpolate(samples.std(0, keepdim=True), size=gt.shape, mode="bilinear")[0, 0].cpu().numpy()
        panels.append(("unc", unc))
        for c, p in enumerate(panels):
            ax = axes[r][c]
            ax.axis("off")
            if isinstance(p, tuple) and isinstance(p[0], str):
                ax.imshow(p[1], cmap="magma", vmin=0, vmax=0.5)
            elif isinstance(p, tuple):
                ax.imshow(p[0])
                ax.text(0.03, 0.04, p[1], transform=ax.transAxes, color="white", fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.6, pad=1.5, linewidth=0))
            else:
                ax.imshow(p)
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
        axes[r][0].text(-0.05, 0.5, note, transform=axes[r][0].transAxes, rotation=90, ha="right", va="center",
                        fontsize=8)
    fig.suptitle(f"{ds}: green = ground truth, red = prediction", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def to_md(t):
    lines = ["| config | " + " | ".join(t.columns) + " |", "|" + "---|" * (len(t.columns) + 1)]
    for c, row in t.iterrows():
        lines.append(f"| {CONFIGS.get(c, (c,))[0]} | " + " | ".join(f"{v:.3f}" for v in row.values) + " |")
    return "\n".join(lines) + "\n"


def main():
    stats = pd.read_csv("results/dataset_stats.csv")
    md, size_tables = ["# Error analysis (mean Dice, seed-averaged per image)\n"], {}
    for ds in ["CVC-ColonDB", "ETIS-LaribPolypDB"]:
        res = bucket_tables(ds, stats)
        if res is None:
            continue
        df, tabs = res
        size_tables[ds] = tabs["size"]
        md += [f"\n## {ds}: by polyp size (area fraction)\n", to_md(tabs["size"]),
               f"\n## {ds}: by polyp/background contrast (ΔE tertiles)\n", to_md(tabs["contrast"])]
        if ds == "CVC-ColonDB" and PROPOSED in df.config.values:
            prop = df[df.config == PROPOSED].sort_values("dice")
            worst = prop.head(15)[["image", "dice", "bf", "area_frac", "contrast_dE"]]
            worst.to_csv("results/worst_cases_colondb.csv", index=False)
            md += ["\n## CVC-ColonDB: 15 worst images for YOLO-Diff (boundary)\n",
                   "| image | dice | bf | area % | contrast ΔE |\n|---|---|---|---|---|\n"]
            md += [f"| {r.image} | {r.dice:.3f} | {r.bf:.3f} | {100 * r.area_frac:.2f} | {r.contrast_dE:.1f} |\n"
                   for r in worst.itertuples()]
            md.append(f"\nFailures (Dice < 0.5): {(prop.dice < 0.5).mean() * 100:.1f}% of images; median area of "
                      f"failures {100 * prop[prop.dice < 0.5].area_frac.median():.2f}% vs "
                      f"{100 * prop[prop.dice >= 0.5].area_frac.median():.2f}% for successes.\n")
            # Qualitative: 2 successes, 2 median cases, 3 worst failures.
            n = len(prop)
            picks = [(prop.iloc[-1].image, "best"), (prop.iloc[int(0.75 * n)].image, "upper quartile"),
                     (prop.iloc[n // 2].image, "median"), (prop.iloc[int(0.25 * n)].image, "lower quartile"),
                     (prop.iloc[2].image, "failure"), (prop.iloc[1].image, "failure"), (prop.iloc[0].image, "worst")]
            qualitative(ds, picks, "results/qualitative_colondb.png")
    if size_tables:
        plot_size(size_tables, "results/dice_by_size.png")
    open("results/error_analysis.md", "w").write("".join(md))
    print("".join(md))


if __name__ == "__main__":
    main()
