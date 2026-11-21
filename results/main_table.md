# Results (mean ± std over seeds; HD95 in pixels at original resolution)

## Seen (Kvasir + ClinicDB test, 162 images pooled)

| Method | Seen dice | Seen iou | Seen bf | Seen hd95 |
|---|---|---|---|---|
| U-Net | 0.844 ± 0.002 | 0.770 ± 0.001 | 0.673 ± 0.009 | 64.21 ± 4.33 |
| PraNet | 0.903 ± 0.005 | 0.852 ± 0.005 | 0.815 ± 0.003 | 37.13 ± 3.09 |
| YOLO-Seg (no diffusion) | 0.910 ± 0.001 | 0.860 ± 0.001 | 0.832 ± 0.003 | 33.04 ± 0.31 |
| YOLO-Diff (no boundary) | 0.849 ± 0.006 | 0.777 ± 0.007 | 0.725 ± 0.009 | 53.79 ± 6.04 |
| YOLO-Diff (boundary) | 0.861 ± 0.007 | 0.797 ± 0.005 | 0.761 ± 0.007 | 47.60 ± 2.73 |
| YOLO-Diff (boundary + strong aug) | 0.862 ± 0.001 | 0.795 ± 0.001 | 0.752 ± 0.006 | 44.48 ± 2.38 |

## Kvasir

| Method | Kvasir dice | Kvasir iou | Kvasir bf | Kvasir hd95 |
|---|---|---|---|---|
| U-Net | 0.849 ± 0.003 | 0.772 ± 0.004 | 0.644 ± 0.012 | 80.64 ± 6.68 |
| PraNet | 0.897 ± 0.006 | 0.844 ± 0.007 | 0.796 ± 0.007 | 52.87 ± 5.62 |
| YOLO-Seg (no diffusion) | 0.906 ± 0.004 | 0.855 ± 0.006 | 0.813 ± 0.008 | 45.68 ± 2.44 |
| YOLO-Diff (no boundary) | 0.864 ± 0.006 | 0.793 ± 0.009 | 0.732 ± 0.005 | 64.08 ± 6.35 |
| YOLO-Diff (boundary) | 0.876 ± 0.001 | 0.814 ± 0.002 | 0.771 ± 0.005 | 57.76 ± 1.52 |
| YOLO-Diff (boundary + strong aug) | 0.877 ± 0.005 | 0.813 ± 0.005 | 0.762 ± 0.010 | 54.06 ± 3.11 |

## CVC-ClinicDB

| Method | CVC-ClinicDB dice | CVC-ClinicDB iou | CVC-ClinicDB bf | CVC-ClinicDB hd95 |
|---|---|---|---|---|
| U-Net | 0.835 ± 0.010 | 0.768 ± 0.007 | 0.719 ± 0.003 | 37.71 ± 6.59 |
| PraNet | 0.914 ± 0.003 | 0.865 ± 0.004 | 0.847 ± 0.005 | 11.74 ± 1.12 |
| YOLO-Seg (no diffusion) | 0.917 ± 0.008 | 0.867 ± 0.010 | 0.862 ± 0.011 | 12.64 ± 3.38 |
| YOLO-Diff (no boundary) | 0.825 ± 0.007 | 0.752 ± 0.004 | 0.715 ± 0.015 | 37.20 ± 6.24 |
| YOLO-Diff (boundary) | 0.837 ± 0.020 | 0.769 ± 0.016 | 0.745 ± 0.015 | 31.22 ± 4.71 |
| YOLO-Diff (boundary + strong aug) | 0.836 ± 0.007 | 0.766 ± 0.006 | 0.737 ± 0.003 | 29.02 ± 1.79 |

## CVC-ColonDB

| Method | CVC-ColonDB dice | CVC-ColonDB iou | CVC-ColonDB bf | CVC-ColonDB hd95 |
|---|---|---|---|---|
| U-Net | 0.649 ± 0.016 | 0.562 ± 0.016 | 0.515 ± 0.018 | 157.57 ± 19.91 |
| PraNet | 0.747 ± 0.009 | 0.676 ± 0.009 | 0.630 ± 0.009 | 109.30 ± 11.72 |
| YOLO-Seg (no diffusion) | 0.777 ± 0.004 | 0.699 ± 0.006 | 0.656 ± 0.013 | 74.22 ± 3.75 |
| YOLO-Diff (no boundary) | 0.714 ± 0.014 | 0.630 ± 0.012 | 0.591 ± 0.007 | 103.47 ± 16.84 |
| YOLO-Diff (boundary) | 0.740 ± 0.007 | 0.655 ± 0.005 | 0.624 ± 0.002 | 87.61 ± 5.34 |
| YOLO-Diff (boundary + strong aug) | 0.754 ± 0.008 | 0.666 ± 0.008 | 0.621 ± 0.003 | 76.45 ± 2.73 |

## ETIS-LaribPolypDB

| Method | ETIS-LaribPolypDB dice | ETIS-LaribPolypDB iou | ETIS-LaribPolypDB bf | ETIS-LaribPolypDB hd95 |
|---|---|---|---|---|
| U-Net | 0.492 ± 0.022 | 0.417 ± 0.016 | 0.383 ± 0.020 | 467.02 ± 61.42 |
| PraNet | 0.703 ± 0.020 | 0.631 ± 0.015 | 0.615 ± 0.010 | 284.05 ± 50.79 |
| YOLO-Seg (no diffusion) | 0.697 ± 0.022 | 0.620 ± 0.023 | 0.614 ± 0.032 | 224.21 ± 11.91 |
| YOLO-Diff (no boundary) | 0.666 ± 0.007 | 0.582 ± 0.007 | 0.575 ± 0.016 | 229.53 ± 21.87 |
| YOLO-Diff (boundary) | 0.683 ± 0.007 | 0.599 ± 0.014 | 0.600 ± 0.009 | 224.45 ± 17.65 |
| YOLO-Diff (boundary + strong aug) | 0.700 ± 0.020 | 0.615 ± 0.021 | 0.603 ± 0.026 | 212.99 ± 19.91 |

## CVC-300

| Method | CVC-300 dice | CVC-300 iou | CVC-300 bf | CVC-300 hd95 |
|---|---|---|---|---|
| U-Net | 0.813 ± 0.016 | 0.730 ± 0.018 | 0.692 ± 0.033 | 62.08 ± 16.73 |
| PraNet | 0.884 ± 0.007 | 0.814 ± 0.010 | 0.792 ± 0.014 | 24.07 ± 8.20 |
| YOLO-Seg (no diffusion) | 0.880 ± 0.014 | 0.811 ± 0.015 | 0.791 ± 0.016 | 22.65 ± 3.69 |
| YOLO-Diff (no boundary) | 0.859 ± 0.005 | 0.784 ± 0.009 | 0.771 ± 0.016 | 32.17 ± 1.68 |
| YOLO-Diff (boundary) | 0.860 ± 0.017 | 0.788 ± 0.017 | 0.769 ± 0.016 | 27.56 ± 6.37 |
| YOLO-Diff (boundary + strong aug) | 0.855 ± 0.006 | 0.784 ± 0.005 | 0.766 ± 0.013 | 28.66 ± 1.76 |

## Model size and speed (batch 1, 352x352, fp32, shared GPU)

| Method | Seeds | Params (M) | Backbone (M) | Latency (ms) |
|---|---|---|---|---|
| U-Net | 3 | 31.04 | 0.00 | 5.1 ± 0.0 |
| PraNet | 3 | 30.48 | 23.65 | 18.6 ± 0.1 |
| YOLO-Seg (no diffusion) | 3 | 14.51 | 5.08 | 18.6 ± 0.2 |
| YOLO-Diff (no boundary) | 3 | 14.51 | 5.08 | 140.0 ± 0.4 |
| YOLO-Diff (boundary) | 3 | 14.51 | 5.08 | 140.0 ± 1.6 |
| YOLO-Diff (boundary + strong aug) | 3 | 14.51 | 5.08 | 140.3 ± 1.5 |
