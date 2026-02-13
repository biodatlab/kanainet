"""
MICCAI dataset loader with pre-defined JSON splits.

Expected directory layout:
    root_dir/
        images/   *.jpg
        masks/    *.png   (same base-name as image)

Split JSON format:
    {"train": ["img1.jpg", ...], "val": [...], "test": [...]}
"""
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .kvasir_dataset import get_val_transform


def get_miccai_train_transform(image_size=(512, 512)):
    """
    Training augmentation for MICCAI endoscopy data.

    Geometric: HFlip, VFlip, RandomRotate90, ShiftScaleRotate,
               random scale 80-120%.
    Motion:    MotionBlur (kernel 3-9).
    Dropout:   CoarseDropout (up to 8 holes, 8×8 – 32×32 px).
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),

        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.0,        # scaling handled separately below
            rotate_limit=15,
            p=0.5,
        ),
        A.RandomScale(scale_limit=(-0.2, 0.2), p=0.1),  # 80–120 %
        # Re-pin to target size after any scale change
        A.Resize(image_size[0], image_size[1]),

        # Motion blur
        A.MotionBlur(blur_limit=(3, 9), p=0.2),

        # Coarse dropout (albumentations ≥ 1.4 API)
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3,
        ),

        # Normalise to ImageNet stats (ConvNeXt pretrained)
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})


class MICCAIDataset(Dataset):
    def __init__(self, root_dir, file_list, image_size=(512, 512),
                 transform=None, mode='train'):
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir  = os.path.join(root_dir, 'masks')
        self.image_size = image_size
        self.transform  = transform
        self.mode       = mode

        # Verify every listed file exists; warn on missing
        self.file_list = []
        for fname in file_list:
            img_path = os.path.join(self.image_dir, fname)
            base = os.path.splitext(fname)[0]
            mask_path = os.path.join(self.mask_dir, base + '.png')
            if not os.path.exists(img_path):
                print(f"[WARN] Missing image: {img_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"[WARN] Missing mask: {mask_path}")
                continue
            self.file_list.append(fname)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        base  = os.path.splitext(fname)[0]

        image = cv2.imread(os.path.join(self.image_dir, fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.mask_dir, base + '.png'),
                          cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            image = cv2.resize(image, self.image_size)
            mask  = cv2.resize(mask, self.image_size,
                               interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask, fname


def create_miccai_loaders(root_dir, split_json, batch_size=4,
                          image_size=(512, 512), num_workers=4):
    """
    Build train / val DataLoaders from a pre-defined JSON split.

    Args:
        root_dir:    Path to MICCAI_data directory.
        split_json:  Path to JSON file with 'train' and 'val' keys.
        batch_size:  Batch size.
        image_size:  (H, W) tuple.
        num_workers: DataLoader workers.

    Returns:
        train_loader, val_loader
    """
    with open(split_json) as f:
        splits = json.load(f)

    train_ds = MICCAIDataset(
        root_dir, splits['train'],
        image_size=image_size,
        transform=get_miccai_train_transform(image_size),
        mode='train'
    )
    val_ds = MICCAIDataset(
        root_dir, splits['val'],
        image_size=image_size,
        transform=get_val_transform(image_size),
        mode='val'
    )

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
