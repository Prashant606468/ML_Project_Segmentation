"""Train one model on Kvasir-SEG + CVC-ClinicDB (PraNet split); select the
checkpoint on a held-out 10% of the training images (never on test data).

    python train.py --model yolo_diff --seed 0
    python train.py --model pranet --seed 0
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from polypseg.data import TrainSet, make_split
from polypseg.metrics import dice_iou
from polypseg.models import MODELS, build_model

# Per-model optimisation defaults: (epochs, lr, backbone lr multiplier)
DEFAULTS = {
    "unet": (100, 1e-3, 1.0),
    "pranet": (50, 1e-4, 1.0),  # official recipe: Adam 1e-4
    "yolo_seg": (100, 5e-4, 0.2),
    "yolo_diff": (100, 5e-4, 0.2),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--no_boundary", action="store_true", help="YOLO models: disable boundary guidance")
    p.add_argument("--yolo", default="s", choices=["n", "s"], help="YOLOv8 backbone size")
    p.add_argument("--aug", default="basic", choices=["basic", "strong"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_every", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--data", default="data")
    p.add_argument("--name", help="run name (default: derived from arguments)")
    return p.parse_args()


def run_name(a):
    name = a.model
    if a.model.startswith("yolo"):
        name += f"_v8{a.yolo}" + ("_nob" if a.no_boundary else "_bnd")
    if a.aug != "basic":
        name += f"_{a.aug}"
    return f"{name}_s{a.seed}"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(0)
    dices = []
    for x, y in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prob = model.predict(x, steps=5, generator=gen) if getattr(model, "diffusion", False) else model.predict(x)
        pred = (prob.float() > 0.5).cpu().numpy()
        dices += [dice_iou(p[0], m[0])[0] for p, m in zip(pred, y.numpy())]
    return float(np.mean(dices))


def main():
    a = parse_args()
    epochs, lr, bb_mult = DEFAULTS[a.model]
    a.epochs, a.lr = a.epochs or epochs, a.lr or lr
    a.name = a.name or run_name(a)
    out = os.path.join("runs", a.name)
    os.makedirs(out, exist_ok=True)
    seed_everything(a.seed)
    device = "cuda"

    train_names, val_names = make_split(a.data, "splits")
    train_dl = DataLoader(TrainSet(a.data, train_names, True, a.aug), a.bs, shuffle=True,
                          num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(TrainSet(a.data, val_names, False), a.bs, num_workers=a.workers)

    model = build_model(a.model, boundary=not a.no_boundary, yolo=a.yolo).to(device)
    bb = list(model.backbone.parameters()) if model.backbone is not None else []
    bb_ids = {id(p) for p in bb}
    head = [p for p in model.parameters() if id(p) not in bb_ids]
    groups = [{"params": head, "lr": a.lr}] + ([{"params": bb, "lr": a.lr * bb_mult}] if bb else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    total = a.epochs * len(train_dl)
    warm = 2 * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / (total - warm))))

    n_params = sum(p.numel() for p in model.parameters())
    cfg = dict(vars(a), bb_lr_mult=bb_mult, n_params=n_params, n_train=len(train_names), n_val=len(val_names))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"[{a.name}] params={n_params / 1e6:.2f}M train={len(train_names)} val={len(val_names)}")

    log = open(os.path.join(out, "log.csv"), "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "loss", "val_dice", "time"])
    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model.compute_loss(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = validate(model, val_dl, device) if ep % a.val_every == 0 or ep == a.epochs else float("nan")
        if val > best:
            best = val
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        writer.writerow([ep, f"{np.mean(losses):.4f}", f"{val:.4f}", f"{time.time() - t0:.1f}"])
        log.flush()
        print(f"[{a.name}] ep {ep:3d} loss {np.mean(losses):.4f} val_dice {val:.4f} best {best:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), os.path.join(out, "last.pt"))
    json.dump({"best_val_dice": best}, open(os.path.join(out, "train_done.json"), "w"))


if __name__ == "__main__":
    main()
