"""Shared plotting style and display names (fixed colour per configuration)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIGS = {  # run-name prefix -> (label, colour); colours follow a fixed categorical order
    "unet": ("U-Net", "#2a78d6"),
    "pranet": ("PraNet", "#eb6834"),
    "yolo_seg_v8s_bnd": ("YOLO-Seg (no diffusion)", "#1baf7a"),
    "yolo_diff_v8s_nob": ("YOLO-Diff (no boundary)", "#eda100"),
    "yolo_diff_v8s_bnd": ("YOLO-Diff (boundary)", "#e87ba4"),
    "yolo_diff_v8s_bnd_strong": ("YOLO-Diff (boundary + strong aug)", "#008300"),
}
DATASETS = ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })
    return plt


def config_of(run):
    return run.rsplit("_s", 1)[0]
