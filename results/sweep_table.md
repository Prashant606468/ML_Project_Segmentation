# YOLO-Diff (boundary): effect of DDIM steps and sample ensembling

| Sampling | Seeds | ColonDB Dice | ColonDB BF | Seen Dice | Latency (ms) |
|---|---|---|---|---|---|
| 1 step | 3 | 0.743 ± 0.007 | 0.625 ± 0.005 | 0.865 ± 0.006 | 17.7 |
| 3 steps | 3 | 0.742 ± 0.006 | 0.625 ± 0.004 | 0.863 ± 0.006 | 45.3 |
| 5 steps | 3 | 0.741 ± 0.006 | 0.630 ± 0.001 | 0.862 ± 0.006 | 71.7 |
| 10 steps (default) | 3 | 0.740 ± 0.007 | 0.624 ± 0.002 | 0.861 ± 0.007 | 140.0 |
| 20 steps | 3 | 0.736 ± 0.007 | 0.614 ± 0.005 | 0.859 ± 0.008 | 254.5 |
| 10 steps x 5 samples | 3 | 0.745 ± 0.006 | 0.639 ± 0.001 | 0.866 ± 0.009 | 137.5 |
