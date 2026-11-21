"""Datasets, splits and augmentation.

Data layout (PraNet benchmark, https://github.com/DengPingFan/PraNet):
    data/TrainDataset/{image,mask}/            900 Kvasir-SEG + 550 CVC-ClinicDB
    data/TestDataset/<name>/{images,masks}/    Kvasir, CVC-ClinicDB, CVC-ColonDB, ETIS-LaribPolypDB, CVC-300

Images are returned in [0, 1]; each model applies its own backbone normalization.
"""
import os
import random
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 352
IN_DOMAIN = ["Kvasir", "CVC-ClinicDB"]
UNSEEN = ["CVC-ColonDB", "ETIS-LaribPolypDB", "CVC-300"]


def source_of(name):
    """Kvasir-SEG files are hash-like ids ('cju...'); CVC-ClinicDB files are numbered."""
    return "CVC-ClinicDB" if os.path.splitext(name)[0].isdigit() else "Kvasir"


def read_image(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def read_mask(path):
    # Some masks are 3-channel or JPEG-like with soft values -> binarize.
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8)


def make_split(data_root, split_dir, val_frac=0.1, seed=0):
    """Fixed train/val split of TrainDataset, stratified by source dataset.

    The split is generated once and stored, so every model and seed uses the
    same images. Validation is only used for checkpoint selection.
    """
    os.makedirs(split_dir, exist_ok=True)
    tr_file, va_file = os.path.join(split_dir, "train.txt"), os.path.join(split_dir, "val.txt")
    if os.path.exists(tr_file) and os.path.exists(va_file):
        return [l.strip() for l in open(tr_file)], [l.strip() for l in open(va_file)]
    names = sorted(os.listdir(os.path.join(data_root, "TrainDataset", "image")))
    rng = random.Random(seed)
    train, val = [], []
    for src in IN_DOMAIN:
        group = [n for n in names if source_of(n) == src]
        rng.shuffle(group)
        k = int(round(val_frac * len(group)))
        val += sorted(group[:k])
        train += sorted(group[k:])
    with open(tr_file, "w") as f:
        f.write("\n".join(train) + "\n")
    with open(va_file, "w") as f:
        f.write("\n".join(val) + "\n")
    return train, val


def get_transform(train, aug="basic", size=IMG_SIZE):
    # A.Resize uses nearest-neighbour for masks, keeping labels binary.
    if not train:
        return A.Compose([A.Resize(size, size)])
    t = [
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.7),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    ]
    if aug == "strong":
        # Appearance augmentation aimed at colour / illumination shift between centres.
        t += [
            A.HueSaturationValue(10, 20, 15, p=0.5),
            A.RandomGamma((80, 120), p=0.3),
            A.OneOf([A.GaussianBlur((3, 5)), A.MotionBlur(5), A.GaussNoise((5.0, 25.0))], p=0.3),
            A.CLAHE(p=0.1),
        ]
    return A.Compose(t)


def to_tensor(img, mask):
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    y = torch.from_numpy(mask).unsqueeze(0).float()
    return x, y


class TrainSet(Dataset):
    def __init__(self, data_root, names, train=True, aug="basic"):
        self.img_dir = os.path.join(data_root, "TrainDataset", "image")
        self.mask_dir = os.path.join(data_root, "TrainDataset", "mask")
        self.names = names
        self.tf = get_transform(train, aug)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = read_image(os.path.join(self.img_dir, n))
        mask = read_mask(os.path.join(self.mask_dir, n))
        out = self.tf(image=img, mask=mask)
        return to_tensor(out["image"], out["mask"])


class TestSet(Dataset):
    """Returns the resized image plus the mask at its original resolution."""

    def __init__(self, data_root, name):
        d = os.path.join(data_root, "TestDataset", name)
        self.images = sorted(glob(os.path.join(d, "images", "*")))
        self.masks = sorted(glob(os.path.join(d, "masks", "*")))
        assert len(self.images) == len(self.masks) > 0, d
        self.tf = get_transform(False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = read_image(self.images[i])
        mask = read_mask(self.masks[i])
        x = torch.from_numpy(self.tf(image=img)["image"]).permute(2, 0, 1).float() / 255.0
        return x, mask, os.path.basename(self.images[i])


def test_collate(batch):
    xs, masks, names = zip(*batch)
    return torch.stack(xs), list(masks), list(names)
