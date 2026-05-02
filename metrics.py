import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import segmentation_models_pytorch as smp

from dataset import ICHDataset, val_transform

IMAGE_DIR = "/home/shashwath/Pleural_Effusion/seg/data/image"
MASK_DIR  = "/home/shashwath/Pleural_Effusion/seg/data/label"
EXP_BASE  = "/home/shashwath/Pleural_Effusion/seg/experiments"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_ids = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")])
_, temp = train_test_split(all_ids, test_size=0.30, random_state=42)
_, test_ids = train_test_split(temp, test_size=0.50, random_state=42)

test_loader = DataLoader(ICHDataset(test_ids, IMAGE_DIR, MASK_DIR, val_transform),
                         batch_size=16, shuffle=False, num_workers=4)


def load_model(exp_name):
    if "unetplusplus" in exp_name:
        encoder = "resnet34" if "pretrained" in exp_name else "resnet18"
        weights = "imagenet" if "pretrained" in exp_name else None
        model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1)
    else:
        encoder = "resnet34" if "pretrained" in exp_name else "resnet18"
        weights = "imagenet" if "pretrained" in exp_name else None
        attn = "scse" if "attention" in exp_name else None
        model = smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1, decoder_attention_type=attn)
    ckpt = os.path.join(EXP_BASE, exp_name, "weights", "best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def compute_metrics(model):
    all_preds, all_masks = [], []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            preds = (torch.sigmoid(model(images)) > 0.5).float().cpu().numpy().flatten()
            all_preds.append(preds)
            all_masks.append(masks.numpy().flatten())

    preds = np.concatenate(all_preds)
    masks = np.concatenate(all_masks)

    inter = (preds * masks).sum()
    dice  = (2 * inter + 1e-6) / (preds.sum() + masks.sum() + 1e-6)
    union = preds.sum() + masks.sum() - inter
    iou   = (inter + 1e-6) / (union + 1e-6)

    return {
        "dice":        dice,
        "iou":         iou,
        "precision":   precision_score(masks, preds, zero_division=0),
        "recall":      recall_score(masks, preds, zero_division=0),
        "f1":          f1_score(masks, preds, zero_division=0),
        "accuracy":    (preds == masks).mean(),
        "specificity": ((preds == 0) & (masks == 0)).sum() / (((preds == 0) & (masks == 0)).sum() + ((preds == 1) & (masks == 0)).sum() + 1e-6),
    }


experiments = [
    "01_vanilla_unet",
    "02_attention_unet",
    "03_unetplusplus",
    "04_vanilla_unet_pretrained",
    "05_attention_unet_pretrained",
    "06_unetplusplus_pretrained",
]

print("\n" + "=" * 90)
print(f"{'experiment':<35} {'dice':>6} {'iou':>6} {'prec':>6} {'recall':>6} {'f1':>6} {'acc':>6} {'spec':>6}")
print("=" * 90)
for exp in experiments:
    if not os.path.exists(os.path.join(EXP_BASE, exp, "weights", "best.pth")):
        print(f"{exp:<35} no weights")
        continue
    m = compute_metrics(load_model(exp))
    print(f"{exp:<35} {m['dice']:>6.4f} {m['iou']:>6.4f} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['f1']:>6.4f} {m['accuracy']:>6.4f} {m['specificity']:>6.4f}")
print("=" * 90)