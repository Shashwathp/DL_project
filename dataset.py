import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 256

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)


class ICHDataset(Dataset):
    def __init__(self, image_ids, image_dir, mask_dir, transform=None):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fname = self.image_ids[idx]
        image = np.array(Image.open(os.path.join(self.image_dir, fname)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, fname)).convert("L"))
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"].unsqueeze(0)
        return image, mask