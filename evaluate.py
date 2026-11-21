"""Evaluate trained runs on all test sets (in-domain and unseen).

Writes, per run:
    runs/<run>/eval/<dataset>.csv   per-image Dice / IoU / BF / HD95
    runs/<run>/eval/summary.json    dataset means + params + latency

    python evaluate.py runs/yolo_diff_v8s_bnd_s0 runs/pranet_s0
    python evaluate.py runs/* --save_preds
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from polypseg.data import IN_DOMAIN, UNSEEN, TestSet, test_collate
from polypseg.metrics import all_metrics
from polypseg.models import build_model


def load_run(run, device):
    cfg = json.load(open(os.path.join(run, "config.json")))
    model = build_model(cfg["model"], boundary=not cfg["no_boundary"], yolo=cfg["yolo"]).to(device)
    model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location=device))
    return model.eval(), cfg


def predict(model, x, steps, n_samples, gen):
    if getattr(model, "diffusion", False):
        return model.predict(x, steps=steps, n_samples=n_samples, generator=gen)
    return model.predict(x)


@torch.no_grad()
def measure_latency(model, steps, n_samples, device, n=50, warmup=10):
    """Mean forward time for one 352x352 image (batch 1), fp32."""
    x = torch.rand(1, 3, 352, 352, device=device)
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(warmup):
        predict(model, x, steps, n_samples, gen)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        predict(model, x, steps, n_samples, gen)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--data", default="data")
    p.add_argument("--steps", type=int, default=10, help="DDIM steps (diffusion models)")
    p.add_argument("--n_samples", type=int, default=1, help="diffusion samples averaged per image")
    p.add_argument("--tag", default="", help="suffix for the eval folder, e.g. for step sweeps")
    p.add_argument("--save_preds", action="store_true")
    p.add_argument("--datasets", nargs="+", default=IN_DOMAIN + UNSEEN)
    a = p.parse_args()
    device = "cuda"

    for run in a.runs:
        if not os.path.exists(os.path.join(run, "best.pt")):
            print(f"skip {run}: no best.pt")
            continue
        model, cfg = load_run(run, device)
        out = os.path.join(run, "eval" + (f"_{a.tag}" if a.tag else ""))
        os.makedirs(out, exist_ok=True)
        summary = {"run": os.path.basename(run.rstrip("/")), "model": cfg["model"], "steps": a.steps,
                   "n_samples": a.n_samples,
                   "params_M": sum(q.numel() for q in model.parameters()) / 1e6,
                   "backbone_params_M": (sum(q.numel() for q in model.backbone.parameters()) / 1e6
                                         if model.backbone is not None else 0.0)}
        for ds in a.datasets:
            gen = torch.Generator(device=device).manual_seed(0)
            dl = DataLoader(TestSet(a.data, ds), batch_size=8, num_workers=4, collate_fn=test_collate)
            rows = []
            for x, masks, names in dl:
                prob = predict(model, x.to(device), a.steps, a.n_samples, gen)
                for pr, m, n in zip(prob, masks, names):
                    pr = F.interpolate(pr[None], size=m.shape, mode="bilinear", align_corners=False)[0, 0]
                    pred = (pr > 0.5).cpu().numpy()
                    rows.append({"image": n, **all_metrics(pred, m)})
                    if a.save_preds:
                        os.makedirs(os.path.join(out, "preds", ds), exist_ok=True)
                        cv2.imwrite(os.path.join(out, "preds", ds, n), (pr.cpu().numpy() * 255).astype(np.uint8))
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(out, f"{ds}.csv"), index=False)
            summary[ds] = df[["dice", "iou", "bf", "hd95"]].mean().round(4).to_dict()
            print(f"{summary['run']:32s} {ds:18s} " + " ".join(f"{k}={v:.4f}" for k, v in summary[ds].items()),
                  flush=True)
        summary["latency_ms"] = measure_latency(model, a.steps, a.n_samples, device)
        print(f"{summary['run']:32s} params={summary['params_M']:.2f}M latency={summary['latency_ms']:.1f}ms")
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
