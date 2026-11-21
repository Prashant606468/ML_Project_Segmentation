from .pranet import PraNet
from .unet import UNet
from .yolo_diff import YoloDiff

MODELS = ["unet", "pranet", "yolo_seg", "yolo_diff"]


def build_model(name, boundary=True, yolo="s", **kw):
    if name == "unet":
        return UNet()
    if name == "pranet":
        return PraNet()
    weights = f"weights/yolov8{yolo}.pt"
    if name == "yolo_seg":
        return YoloDiff(weights, diffusion=False, boundary=boundary, **kw)
    if name == "yolo_diff":
        return YoloDiff(weights, diffusion=True, boundary=boundary, **kw)
    raise ValueError(name)
