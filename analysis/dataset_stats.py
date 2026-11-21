"""Domain-gap diagnosis: how do the datasets differ in colour, polyp size and contrast?

Outputs: results/dataset_stats.csv (per image), results/dataset_stats.md (medians),
         results/dataset_stats.png
"""
import os
import sys
from glob import glob

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from polypseg.data import read_image, read_mask, source_of  # noqa: E402
from style import setup  # noqa: E402

DATA = "data"


def image_stats(img, mask):
    """Colour and lesion statistics. Black endoscope borders are excluded."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255  # L in [0, 100]; a, b offset by 128
    lab[..., 1:] -= 128
    valid = img.max(2) > 20
    m = mask.astype(bool)
    ring = cv2.dilate(mask, np.ones((31, 31), np.uint8)).astype(bool) & ~m & valid
    n_cc = cv2.connectedComponents(mask)[0] - 1
    contrast = np.linalg.norm(lab[m].mean(0) - lab[ring].mean(0)) if m.any() and ring.any() else np.nan
    return {"H": img.shape[0], "W": img.shape[1],
            "L": lab[valid][:, 0].mean(), "a": lab[valid][:, 1].mean(), "b": lab[valid][:, 2].mean(),
            "area_frac": m.sum() / valid.sum(), "n_polyps": n_cc, "contrast_dE": contrast}


def collect():
    rows = []
    tr = os.path.join(DATA, "TrainDataset")
    for n in sorted(os.listdir(os.path.join(tr, "image"))):
        s = image_stats(read_image(os.path.join(tr, "image", n)), read_mask(os.path.join(tr, "mask", n)))
        rows.append({"dataset": f"{source_of(n)} (train)", "image": n, **s})
    for d in sorted(glob(os.path.join(DATA, "TestDataset", "*"))):
        name = os.path.basename(d)
        for ip, mp in zip(sorted(glob(f"{d}/images/*")), sorted(glob(f"{d}/masks/*"))):
            s = image_stats(read_image(ip), read_mask(mp))
            rows.append({"dataset": f"{name} (test)", "image": os.path.basename(ip), **s})
    return pd.DataFrame(rows)


ORDER = ["Kvasir (train)", "CVC-ClinicDB (train)", "Kvasir (test)", "CVC-ClinicDB (test)",
         "CVC-ColonDB (test)", "ETIS-LaribPolypDB (test)", "CVC-300 (test)"]


def main():
    os.makedirs("results", exist_ok=True)
    df = collect()
    df.to_csv("results/dataset_stats.csv", index=False)
    g = df.groupby("dataset")
    summ = pd.DataFrame({
        "images": g.size(),
        "resolution (median HxW)": g.apply(lambda d: f"{int(d.H.median())}x{int(d.W.median())}"),
        "L* mean": g.L.mean().astype(float).round(1), "a* mean": g.a.mean().astype(float).round(1), "b* mean": g.b.mean().astype(float).round(1),
        "polyp area % (median)": (100 * g.area_frac.median()).round(2),
        "small polyps (<2% area) %": (100 * g.apply(lambda d: (d.area_frac < 0.02).mean())).round(1),
        "multi-polyp images %": (100 * g.apply(lambda d: (d.n_polyps > 1).mean())).round(1),
        "polyp/background contrast ΔE (median)": g.contrast_dE.median().round(1),
    }).reindex([o for o in ORDER if o in g.groups])
    with open("results/dataset_stats.md", "w") as f:
        f.write("# Dataset statistics (domain-gap diagnosis)\n\n")
        cols = ["dataset"] + list(summ.columns)
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for idx, row in summ.iterrows():
            f.write("| " + " | ".join([idx] + [str(v) for v in row.values]) + " |\n")
    print(open("results/dataset_stats.md").read())

    plt = setup()
    panels = [("area_frac", "Polyp area (fraction of image, log)", True), ("contrast_dE", "Polyp/background ΔE", False),
              ("L", "Mean lightness L*", False), ("a", "Mean redness a*", False)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    labels = [o for o in ORDER if o in g.groups]
    for ax, (col, title, log) in zip(axes.flat, panels):
        data = [df[df.dataset == o][col].dropna().values for o in labels]
        bp = ax.boxplot(data, vert=False, labels=labels, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": "#0b0b0b", "linewidth": 2})
        for i, patch in enumerate(bp["boxes"]):
            unseen = "ColonDB" in labels[i] or "ETIS" in labels[i] or "300" in labels[i]
            patch.set_facecolor("#eb6834" if unseen else "#2a78d6")
            patch.set_alpha(0.75)
        if log:
            ax.set_xscale("log")
        ax.set_title(title, loc="left")
        ax.invert_yaxis()
    fig.suptitle("Where the domains differ (blue = training sources, orange = unseen test sets)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig("results/dataset_stats.png", dpi=200)


if __name__ == "__main__":
    main()
