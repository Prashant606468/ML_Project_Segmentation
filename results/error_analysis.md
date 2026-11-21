# Error analysis (mean Dice, seed-averaged per image)

## CVC-ColonDB: by polyp size (area fraction)
| config | small (<2%) (n=79) | medium (2-8%) (n=182) | large (>8%) (n=119) |
|---|---|---|---|
| U-Net | 0.471 | 0.786 | 0.556 |
| PraNet | 0.555 | 0.861 | 0.700 |
| YOLO-Seg (no diffusion) | 0.635 | 0.857 | 0.750 |
| YOLO-Diff (no boundary) | 0.541 | 0.846 | 0.626 |
| YOLO-Diff (boundary) | 0.565 | 0.844 | 0.697 |
| YOLO-Diff (boundary + strong aug) | 0.594 | 0.851 | 0.711 |

## CVC-ColonDB: by polyp/background contrast (ΔE tertiles)
| config | low (n=127) | medium (n=126) | high (n=126) |
|---|---|---|---|
| U-Net | 0.655 | 0.726 | 0.570 |
| PraNet | 0.737 | 0.801 | 0.709 |
| YOLO-Seg (no diffusion) | 0.784 | 0.802 | 0.751 |
| YOLO-Diff (no boundary) | 0.703 | 0.758 | 0.687 |
| YOLO-Diff (boundary) | 0.733 | 0.766 | 0.727 |
| YOLO-Diff (boundary + strong aug) | 0.746 | 0.772 | 0.750 |

## CVC-ColonDB: 15 worst images for YOLO-Diff (boundary)
| image | dice | bf | area % | contrast ΔE |
|---|---|---|---|---|
| 360.png | 0.000 | 0.000 | 0.47 | 1.2 |
| 303.png | 0.000 | 0.022 | 61.36 | 24.7 |
| 313.png | 0.000 | 0.000 | 35.02 | 7.7 |
| 380.png | 0.000 | 0.000 | 1.69 | nan |
| 309.png | 0.000 | 0.000 | 26.94 | 3.2 |
| 357.png | 0.000 | 0.000 | 0.51 | 1.1 |
| 320.png | 0.006 | 0.030 | 34.02 | 10.7 |
| 314.png | 0.007 | 0.000 | 38.61 | 2.4 |
| 315.png | 0.014 | 0.034 | 29.16 | 9.3 |
| 316.png | 0.019 | 0.081 | 21.13 | 8.4 |
| 342.png | 0.026 | 0.086 | 0.35 | 7.3 |
| 353.png | 0.032 | 0.042 | 0.75 | 0.6 |
| 30.png | 0.040 | 0.085 | 4.12 | 28.5 |
| 301.png | 0.043 | 0.065 | 37.76 | 11.0 |
| 352.png | 0.056 | 0.056 | 0.96 | 2.0 |

Failures (Dice < 0.5): 18.4% of images; median area of failures 2.33% vs 4.82% for successes.

## ETIS-LaribPolypDB: by polyp size (area fraction)
| config | small (<2%) (n=94) | medium (2-8%) (n=63) | large (>8%) (n=39) |
|---|---|---|---|
| U-Net | 0.360 | 0.539 | 0.734 |
| PraNet | 0.562 | 0.785 | 0.909 |
| YOLO-Seg (no diffusion) | 0.549 | 0.789 | 0.902 |
| YOLO-Diff (no boundary) | 0.522 | 0.759 | 0.862 |
| YOLO-Diff (boundary) | 0.542 | 0.777 | 0.874 |
| YOLO-Diff (boundary + strong aug) | 0.549 | 0.801 | 0.899 |

## ETIS-LaribPolypDB: by polyp/background contrast (ΔE tertiles)
| config | low (n=66) | medium (n=65) | high (n=65) |
|---|---|---|---|
| U-Net | 0.298 | 0.477 | 0.704 |
| PraNet | 0.518 | 0.701 | 0.892 |
| YOLO-Seg (no diffusion) | 0.525 | 0.700 | 0.867 |
| YOLO-Diff (no boundary) | 0.497 | 0.656 | 0.847 |
| YOLO-Diff (boundary) | 0.523 | 0.671 | 0.859 |
| YOLO-Diff (boundary + strong aug) | 0.547 | 0.683 | 0.871 |
