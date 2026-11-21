"""Aggregate evaluation results over seeds into tables and an efficiency plot.

Outputs (results/):
    main_table.md / main_table.csv   mean ± std over seeds, every dataset and metric
    gap_table.md                     in-domain vs unseen Dice and the generalization gap
    sweep_table.md                   DDIM steps / sample-ensemble sweep (proposed model)
    accuracy_efficiency.png          unseen ColonDB Dice vs latency
"""
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from style import CONFIGS, DATASETS, MUTED, config_of, setup  # noqa: E402

METRICS = ["dice", "iou", "bf", "hd95"]
OUT = "results"


def load(tag=""):
    rows = []
    for f in glob(f"runs/*/eval{('_' + tag) if tag else ''}/summary.json"):
        s = json.load(open(f))
        cfg = config_of(s["run"])
        r = {"config": cfg, "run": s["run"], "params_M": s["params_M"],
             "backbone_params_M": s["backbone_params_M"], "latency_ms": s["latency_ms"]}
        for ds in DATASETS:
            if ds in s:
                r.update({f"{ds}/{m}": s[ds][m] for m in METRICS})
        # Pooled in-domain score: per-image mean over Kvasir + ClinicDB test images (162).
        d = os.path.dirname(f)
        per_img = [pd.read_csv(os.path.join(d, f"{ds}.csv")) for ds in ["Kvasir", "CVC-ClinicDB"]
                   if os.path.exists(os.path.join(d, f"{ds}.csv"))]
        if len(per_img) == 2:
            pooled = pd.concat(per_img)
            r.update({f"Seen/{m}": pooled[m].mean() for m in METRICS})
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(mean, std, digits=3):
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if not np.isnan(std) else f"{mean:.{digits}f}"


def agg(df):
    g = df.groupby("config")
    mean, std, n = g.mean(numeric_only=True), g.std(numeric_only=True), g.size()
    order = [c for c in CONFIGS if c in mean.index] + [c for c in mean.index if c not in CONFIGS]
    return mean.loc[order], std.loc[order], n.loc[order]


def label(c):
    return CONFIGS.get(c, (c,))[0]


def to_md(table):
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.values]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    if df.empty:
        print("no results found")
        return
    mean, std, n = agg(df)

    rows = []
    for c in mean.index:
        r = {"Method": label(c), "Seeds": n[c], "Params (M)": f"{mean.loc[c, 'params_M']:.2f}",
             "Backbone (M)": f"{mean.loc[c, 'backbone_params_M']:.2f}",
             "Latency (ms)": fmt(mean.loc[c, "latency_ms"], std.loc[c, "latency_ms"], 1)}
        for ds in ["Seen"] + DATASETS:
            for m in METRICS:
                k = f"{ds}/{m}"
                if k in mean.columns:
                    r[f"{ds} {m}"] = fmt(mean.loc[c, k], std.loc[c, k], 2 if m == "hd95" else 3)
        rows.append(r)
    main_t = pd.DataFrame(rows)
    main_t.to_csv(f"{OUT}/main_table.csv", index=False)

    with open(f"{OUT}/main_table.md", "w") as f:
        f.write("# Results (mean ± std over seeds; HD95 in pixels at original resolution)\n\n")
        base = ["Method", "Seeds", "Params (M)", "Backbone (M)", "Latency (ms)"]
        for ds in ["Seen"] + DATASETS:
            cols = [c for c in main_t.columns if c.startswith(ds + " ")]
            if cols:
                title = "Seen (Kvasir + ClinicDB test, 162 images pooled)" if ds == "Seen" else ds
                f.write(f"## {title}\n\n" + to_md(main_t[base[:1] + cols]) + "\n")
        f.write("## Model size and speed (batch 1, 352x352, fp32, shared GPU)\n\n" + to_md(main_t[base]))

    # Generalization gap: seen pooled Dice minus unseen Dice, per seed then aggregated.
    gap_rows = []
    for c in mean.index:
        d = df[df.config == c]
        r = {"Method": label(c), "Seen Dice": fmt(d["Seen/dice"].mean(), d["Seen/dice"].std())}
        for ds in ["CVC-ColonDB", "ETIS-LaribPolypDB"]:
            gap = d["Seen/dice"] - d[f"{ds}/dice"]
            r[f"{ds} Dice"] = fmt(d[f"{ds}/dice"].mean(), d[f"{ds}/dice"].std())
            r[f"Gap to {ds}"] = fmt(gap.mean(), gap.std())
        gap_rows.append(r)
    with open(f"{OUT}/gap_table.md", "w") as f:
        f.write("# Cross-dataset generalization gap (Dice; gap = seen − unseen, lower is better)\n\n")
        f.write(to_md(pd.DataFrame(gap_rows)))

    # DDIM step / ensemble sweep for the proposed model.
    sweep = []
    for tag, desc in [("steps1", "1 step"), ("steps3", "3 steps"), ("steps5", "5 steps"), ("", "10 steps (default)"),
                      ("steps20", "20 steps"), ("ens5", "10 steps x 5 samples")]:
        d = load(tag)
        d = d[d.config == "yolo_diff_v8s_bnd"] if not d.empty else d
        if d.empty:
            continue
        sweep.append({"Sampling": desc, "Seeds": len(d),
                      "ColonDB Dice": fmt(d["CVC-ColonDB/dice"].mean(), d["CVC-ColonDB/dice"].std()),
                      "ColonDB BF": fmt(d["CVC-ColonDB/bf"].mean(), d["CVC-ColonDB/bf"].std()),
                      "Seen Dice": fmt(d["Seen/dice"].mean(), d["Seen/dice"].std()),
                      "Latency (ms)": f"{d['latency_ms'].mean():.1f}"})
    if sweep:
        with open(f"{OUT}/sweep_table.md", "w") as f:
            f.write("# YOLO-Diff (boundary): effect of DDIM steps and sample ensembling\n\n")
            f.write(to_md(pd.DataFrame(sweep)))

    # Accuracy vs efficiency: ColonDB Dice against latency; marker area ~ parameters.
    plt = setup()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in mean.index:
        name, color = CONFIGS.get(c, (c, MUTED))
        x, y = mean.loc[c, "latency_ms"], mean.loc[c, "CVC-ColonDB/dice"]
        ax.errorbar(x, y, yerr=std.loc[c, "CVC-ColonDB/dice"], fmt="none", ecolor=color, elinewidth=1.5, capsize=3)
        ax.scatter(x, y, s=12 * mean.loc[c, "params_M"], color=color, edgecolor="#fcfcfb", linewidth=2,
                   label=f"{name} ({mean.loc[c, 'params_M']:.1f}M)", zorder=3)
    ax.set_xlabel("Inference latency per image (ms, batch 1)")
    ax.set_ylabel("Dice on unseen CVC-ColonDB")
    ax.set_title("Accuracy vs. efficiency (marker area ∝ parameters; bars = ±1 std over seeds)", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, markerscale=0.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/accuracy_efficiency.png", dpi=200, bbox_inches="tight")

    print(open(f"{OUT}/main_table.md").read())
    print(open(f"{OUT}/gap_table.md").read())
    if sweep:
        print(open(f"{OUT}/sweep_table.md").read())


if __name__ == "__main__":
    main()
