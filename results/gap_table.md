# Cross-dataset generalization gap (Dice; gap = seen − unseen, lower is better)

| Method | Seen Dice | CVC-ColonDB Dice | Gap to CVC-ColonDB | ETIS-LaribPolypDB Dice | Gap to ETIS-LaribPolypDB |
|---|---|---|---|---|---|
| U-Net | 0.844 ± 0.002 | 0.649 ± 0.016 | 0.195 ± 0.016 | 0.492 ± 0.022 | 0.352 ± 0.022 |
| PraNet | 0.903 ± 0.005 | 0.747 ± 0.009 | 0.156 ± 0.012 | 0.703 ± 0.020 | 0.201 ± 0.021 |
| YOLO-Seg (no diffusion) | 0.910 ± 0.001 | 0.777 ± 0.004 | 0.133 ± 0.004 | 0.697 ± 0.022 | 0.214 ± 0.022 |
| YOLO-Diff (no boundary) | 0.849 ± 0.006 | 0.714 ± 0.014 | 0.136 ± 0.014 | 0.666 ± 0.007 | 0.184 ± 0.007 |
| YOLO-Diff (boundary) | 0.861 ± 0.007 | 0.740 ± 0.007 | 0.121 ± 0.005 | 0.683 ± 0.007 | 0.178 ± 0.007 |
| YOLO-Diff (boundary + strong aug) | 0.862 ± 0.001 | 0.754 ± 0.008 | 0.108 ± 0.008 | 0.700 ± 0.020 | 0.162 ± 0.019 |
