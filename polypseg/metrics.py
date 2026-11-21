"""Per-image segmentation metrics, computed at the original mask resolution."""
import numpy as np
from scipy import ndimage


def _boundary(m):
    m = m.astype(bool)
    if not m.any():
        return m
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)


def dice_iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    ps, gs = p.sum(), g.sum()
    if ps + gs == 0:
        return 1.0, 1.0
    return 2 * inter / (ps + gs), inter / (ps + gs - inter)


def boundary_f(pred, gt, tol_frac=0.0075):
    """Boundary F-score (Csurka et al., BMVC 2013).

    A boundary pixel is matched if it lies within `tol_frac` x image diagonal
    of the other boundary.
    """
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() and not bg.any():
        return 1.0
    if not bp.any() or not bg.any():
        return 0.0
    tol = max(1.0, tol_frac * np.hypot(*gt.shape))
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    precision = (dt_g[bp] <= tol).mean()
    recall = (dt_p[bg] <= tol).mean()
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hd95(pred, gt):
    """Symmetric 95th-percentile Hausdorff distance between boundaries (pixels).

    If exactly one mask is empty the distance is undefined; we return the image
    diagonal as a worst-case penalty so missed polyps are not silently dropped.
    """
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() and not bg.any():
        return 0.0
    if not bp.any() or not bg.any():
        return float(np.hypot(*gt.shape))
    d_pg = ndimage.distance_transform_edt(~bg)[bp]
    d_gp = ndimage.distance_transform_edt(~bp)[bg]
    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95))


def all_metrics(pred, gt):
    d, i = dice_iou(pred, gt)
    return {"dice": float(d), "iou": float(i), "bf": float(boundary_f(pred, gt)), "hd95": hd95(pred, gt)}
