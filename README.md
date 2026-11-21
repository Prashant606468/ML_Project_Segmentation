# Polyp Segmentation using YOLO-Backbone-Guided Diffusion

**Cross-dataset generalization of colorectal polyp segmentation: train on Kvasir-SEG + CVC-ClinicDB, test on the unseen CVC-ColonDB.**

Prashant Kumar Mittra (m24ma2009@iitj.ac.in), Pawan Kumar Koli (m24ma2007@iitj.ac.in), Deep Das (m26mac004@iitj.ac.in), IIT Jodhpur. Machine Learning course project.

We propose a conditional diffusion model whose denoiser is guided by multi-scale features from a lightweight YOLOv8 backbone, and compare it with U-Net and PraNet. The ablations switch off diffusion, boundary guidance and appearance augmentation one at a time.

## Main result (3 seeds, mean ± std)

| Method | Params | Seen Dice (Kvasir+ClinicDB) | **ColonDB Dice** (unseen) | ColonDB BF | ColonDB HD95 ↓ | Latency* |
|---|---|---|---|---|---|---|
| U-Net | 31.0M | 0.844 ± 0.002 | 0.649 ± 0.016 | 0.515 | 157.6 | 5 ms |
| PraNet (baseline) | 30.5M | 0.903 ± 0.005 | 0.747 ± 0.009 | 0.630 | 109.3 | 19 ms |
| **YOLO-Seg** (ours, no diffusion) | **14.5M** | **0.910 ± 0.001** | **0.777 ± 0.004** | **0.656** | **74.2** | 19 ms |
| YOLO-Diff, no boundary | 14.5M | 0.849 ± 0.006 | 0.714 ± 0.014 | 0.591 | 103.5 | 140 ms |
| YOLO-Diff, boundary | 14.5M | 0.861 ± 0.007 | 0.740 ± 0.007 | 0.624 | 87.6 | 140 ms |
| YOLO-Diff, boundary + strong aug | 14.5M | 0.862 ± 0.001 | 0.754 ± 0.008 | 0.621 | 76.5 | 140 ms |

\*Batch 1, 352×352, fp32, on a GPU shared with other users, so absolute latencies are noisy. YOLO-Diff uses 10 DDIM steps.

The YOLOv8s backbone and our decoder beat PraNet on the unseen dataset with half the parameters. The diffusion sampling itself adds no accuracy in our setup: 1 DDIM step scores the same as 20 steps. Within the diffusion model, boundary guidance and appearance augmentation both help. The full tables are in [results/](results/): [main_table.md](results/main_table.md), [gap_table.md](results/gap_table.md), [sweep_table.md](results/sweep_table.md), [error_analysis.md](results/error_analysis.md) and [dataset_stats.md](results/dataset_stats.md).

## Repository structure

```
polypseg/                 library code
  data.py                 dataset loading, fixed train/val split, augmentation
  losses.py               BCE+Dice, PraNet structure loss, edge loss
  metrics.py              Dice, IoU, Boundary F-score, HD95
  models/
    yolo_diff.py          proposed model (YOLOv8 backbone + FPN neck + conditional denoiser + DDIM)
    pranet.py             PraNet (port of the official code)
    unet.py               U-Net
train.py                  train one configuration
evaluate.py               evaluate checkpoints on all test sets (per-image CSV + summary + latency)
analysis/                 tables, domain-gap statistics, error analysis, figures
scripts/
  download_data.sh        datasets + YOLOv8 weights
  checkpoints.py          download / upload trained checkpoints (Hugging Face Hub)
  reproduce_eval.sh       reproduce all numbers and figures from checkpoints
  git_log_table.py        commit table for the report
run_all.sh                full pipeline: 18 training runs -> evaluation -> analysis
splits/                   the fixed train/val split (train.txt, val.txt)
runs/<run>/               per-run config.json, log.csv, eval/*.csv, eval/summary.json (checkpoints not in git)
results/                  final tables and figures
logs/                     training / evaluation logs of the reported runs
report/                   CVPR-format report (LaTeX)
REPRODUCIBILITY.md        reproducibility declaration and artifacts
```

## 1. Environment

Tested on Ubuntu 20.04, Python 3.9.22, PyTorch 2.6.0 + CUDA 12.4, and an NVIDIA RTX 5000 Ada (32 GB).

```bash
conda env create -f environment.yml && conda activate polypseg
# or: pip install -r requirements.txt
```

## 2. Data

We use the PraNet benchmark split ([PraNet repo](https://github.com/DengPingFan/PraNet), Google Drive):

| Set | Images | Role |
|---|---|---|
| TrainDataset | 900 Kvasir-SEG + 550 CVC-ClinicDB | training (90%) / validation (10%, [splits/](splits/)) |
| Kvasir test, CVC-ClinicDB test | 100 + 62 | in-domain ("seen") test |
| CVC-ColonDB | 380 | **unseen** test (main target) |
| ETIS-LaribPolypDB, CVC-300 | 196, 60 | additional unseen test sets |

```bash
bash scripts/download_data.sh      # -> data/TrainDataset, data/TestDataset, weights/yolov8{n,s}.pt
sha256sum -c data_and_weights.sha256
```

The datasets are not in this repository (1.5 GB). The script downloads them from the public PraNet links. The validation split is used **only** for checkpoint selection; test sets are never used for training or model selection.

## 3. Reproduce from released checkpoints (no training, ~30 min)

The checkpoints (18 × 56–119 MB, 1.4 GB in total) exceed GitHub's file limit, so they are hosted on the Hugging Face Hub: **https://huggingface.co/HF_REPO_ID**

```bash
bash scripts/reproduce_eval.sh HF_REPO_ID
```

This downloads and verifies the checkpoints (SHA-256), evaluates all 18 runs, runs the DDIM-step sweep, and regenerates every table and figure in `results/`. To evaluate a single model:

```bash
python scripts/checkpoints.py download --repo HF_REPO_ID --runs yolo_seg_v8s_bnd_s0
python evaluate.py runs/yolo_seg_v8s_bnd_s0
```

## 4. Reproduce from scratch (training, ~6 h on one shared GPU)

```bash
bash run_all.sh
```

This trains 6 configurations × 3 seeds (0, 1, 2), three runs in parallel, then evaluates and analyses them. Single runs:

```bash
python train.py --model unet --seed 0
python train.py --model pranet --seed 0
python train.py --model yolo_seg --seed 0                     # YOLO-Seg (boundary guidance on)
python train.py --model yolo_diff --no_boundary --seed 0      # YOLO-Diff, no boundary guidance
python train.py --model yolo_diff --seed 0                    # YOLO-Diff, boundary guidance
python train.py --model yolo_diff --aug strong --seed 0       # + appearance augmentation
python evaluate.py runs/yolo_diff_v8s_bnd_s0 --steps 10       # DDIM steps; --n_samples for ensembling
```

| Model | Epochs | Optimizer | LR (head / backbone) | Loss |
|---|---|---|---|---|
| U-Net | 100 | AdamW, cosine, 2-epoch warm-up | 1e-3 | BCE + Dice |
| PraNet | 50 | AdamW, cosine | 1e-4 / 1e-4 | structure loss on 4 side outputs (official) |
| YOLO-Seg / YOLO-Diff | 100 | AdamW, cosine | 5e-4 / 1e-4 | structure + 0.5·edge (boundary), or BCE + Dice |

All models: batch 16, 352×352, bf16 autocast, gradient clipping 1.0, identical augmentation, and checkpoint selection on validation Dice (computed every 2 epochs).

## Evaluation protocol

- Predictions are upsampled to the original mask resolution and thresholded at 0.5. Metrics are computed per image and averaged.
- **Boundary F-score**: the tolerance is 0.75% of the image diagonal. **HD95**: symmetric 95th-percentile boundary distance in pixels; a missed polyp is penalised with the image diagonal.
- "Seen" pools the 162 Kvasir + ClinicDB test images.
- Diffusion sampling uses a fixed noise seed, so evaluation is deterministic.

## Notes and limitations

- PraNet is trained with the same augmentation as all other models and without its multi-scale training. It is evaluated with a fixed 0.5 threshold, so its numbers differ slightly from the paper (ours: ColonDB 0.747, paper: 0.709).
- The Res2Net-50 ImageNet weights for PraNet are loaded through `timm` (`res2net50_26w_4s`) because the original download link is unmaintained.
- Latency was measured on a shared GPU. Use it for relative comparisons only.
- The denoiser (9.2M parameters) is larger than the YOLOv8s backbone (5.1M). "Lightweight" refers to the backbone and the total model size.

## References

- D.-P. Fan et al. PraNet: Parallel Reverse Attention Network for Polyp Segmentation. MICCAI 2020. https://github.com/DengPingFan/PraNet
- D. Jha et al. Kvasir-SEG: A Segmented Polyp Dataset. MMM 2020.
- J. Bernal et al. WM-DOVA maps for accurate polyp highlighting in colonoscopy (CVC-ClinicDB). CMIG 2015.
- N. Tajbakhsh et al. Automated polyp detection in colonoscopy videos using shape and context information (CVC-ColonDB). IEEE TMI 2016.
- J. Song et al. DDIM: Denoising Diffusion Implicit Models. ICLR 2021.
- G. Jocher et al. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
- S. Ali et al. PolypGen: A multi-center polyp detection and segmentation dataset. Scientific Data 2023.
