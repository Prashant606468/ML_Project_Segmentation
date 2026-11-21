# Reproducibility Declaration

We, the authors (Prashant Kumar Mittra, Pawan Kumar Koli, Deep Das), declare that:

1. **Every number, table and figure in the report and in `results/` was produced by the code in this repository**, at the commit listed in the report, using the commands in [README.md](README.md). No result was edited by hand.
2. **No test data was used for training, hyper-parameter choice or checkpoint selection.** Checkpoints were selected on a fixed 10% validation split of the training images ([splits/val.txt](splits/val.txt)). CVC-ColonDB, ETIS-LaribPolypDB and CVC-300 were never seen during training.
3. **All configurations were run with 3 seeds (0, 1, 2)**, and all runs are reported (mean ± std). No run was discarded.
4. **The released checkpoints are the exact ones evaluated in the report.** Their SHA-256 digests are in [checkpoints.sha256](checkpoints.sha256).
5. **Baselines were trained under the same protocol** (data split, resolution, augmentation, checkpoint selection) as the proposed model. Deviations from the original PraNet recipe are listed in the README.

## What was verified

| Check | Result |
|---|---|
| Re-evaluating saved checkpoints (`yolo_seg_v8s_bnd_s0`, `yolo_diff_v8s_bnd_s0` on CVC-ColonDB and Kvasir) | **Bit-identical** per-image Dice, IoU, BF and HD95 (max abs. difference 0.0). See `runs/*/eval_repro/` |
| Diffusion sampling | Deterministic: a fixed `torch.Generator` seed (0) is used for the initial noise |
| Seed variability (re-training) | Reported as ± std over 3 seeds. ColonDB Dice std ≤ 0.016 for all methods |

## Known sources of non-determinism

- **Re-training** is not bit-exact. cuDNN convolution and bilinear-upsampling backward kernels are non-deterministic on GPU, and data-loader worker scheduling varies. Re-trained models should fall within the reported seed standard deviations, not match them exactly.
- **Latency** depends on GPU load. Our GPU was shared with other users' jobs. Re-running the same checkpoint measured 11 ms instead of 19 ms (YOLO-Seg) and 80 ms instead of 140 ms (YOLO-Diff). Only relative latencies measured in the same session are meaningful.

## Artifacts

| Artifact | Location |
|---|---|
| Code | this repository (commit history listed in the report) |
| Environment | [requirements.txt](requirements.txt) (exact pinned versions), [environment.yml](environment.yml) |
| Data | public PraNet split, downloaded by [scripts/download_data.sh](scripts/download_data.sh); SHA-256 in [data_and_weights.sha256](data_and_weights.sha256) |
| Pretrained weights | YOLOv8n/s (Ultralytics v8.4.0 release), Res2Net-50 26w_4s (timm); SHA-256 in [data_and_weights.sha256](data_and_weights.sha256) |
| Data split | [splits/train.txt](splits/train.txt) (1305), [splits/val.txt](splits/val.txt) (145), generated with seed 0 |
| Trained checkpoints | Hugging Face Hub (link in README), 18 × `best.pt`, verified by [checkpoints.sha256](checkpoints.sha256) |
| Per-run configuration and training curves | `runs/<run>/config.json`, `runs/<run>/log.csv` |
| Per-image test results | `runs/<run>/eval/<dataset>.csv` |
| Logs of the reported runs | [logs/](logs/) |
| Final tables and figures | [results/](results/) |

## Hardware and software

- GPU: NVIDIA RTX 5000 Ada Generation (32 GB), driver 580.105.08, shared with other users
- CPU: Intel Xeon Silver 4314; OS: Ubuntu 20.04.6 LTS
- Python 3.9.22, PyTorch 2.6.0 (CUDA 12.4, cuDNN 9.1), timm 0.6.12, ultralytics 8.4.24, albumentations 1.3.0
- Training time: ~10–25 s per epoch per run (3 runs in parallel); ~6 h for all 18 runs
