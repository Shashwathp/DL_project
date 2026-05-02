import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from dataset import ICHDataset, train_transform, val_transform
from utils import DiceBCELoss, train_one_epoch, evaluate, save_plots, save_predictions

IMAGE_DIR = "/home/shashwath/Pleural_Effusion/seg/data/image"
MASK_DIR  = "/home/shashwath/Pleural_Effusion/seg/data/label"
EXP_BASE  = "/home/shashwath/Pleural_Effusion/seg/experiments"
EPOCHS     = 50
BATCH_SIZE = 16
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)
print(f"device: {DEVICE}")
print(f"gpu: {torch.cuda.get_device_name(0)}")

all_ids = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")])
train_ids, temp = train_test_split(all_ids, test_size=0.30, random_state=42)
val_ids, test_ids = train_test_split(temp, test_size=0.50, random_state=42)
print(f"train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")

train_loader = DataLoader(ICHDataset(train_ids, IMAGE_DIR, MASK_DIR, train_transform),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(ICHDataset(val_ids, IMAGE_DIR, MASK_DIR, val_transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(ICHDataset(test_ids, IMAGE_DIR, MASK_DIR, val_transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

experiments = [
    ("01_vanilla_unet",               smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)),
    ("02_attention_unet",             smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1, decoder_attention_type="scse")),
    ("03_unetplusplus",               smp.UnetPlusPlus(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)),
    ("04_vanilla_unet_pretrained",    smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)),
    ("05_attention_unet_pretrained",  smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, decoder_attention_type="scse")),
    ("06_unetplusplus_pretrained",    smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)),
]

all_results = {}

for exp_name, model in experiments:
    print(f"\n{exp_name}")
    print("-" * 50)

    w_dir    = os.path.join(EXP_BASE, exp_name, "weights")
    p_dir    = os.path.join(EXP_BASE, exp_name, "plots")
    pred_dir = os.path.join(EXP_BASE, exp_name, "predictions")

    model     = model.to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

    history   = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [], "val_iou": []}
    best_dice = 0
    ckpt_path = os.path.join(w_dir, "best.pth")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl_loss, vl_dice, vl_iou = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(vl_dice)
        history["val_iou"].append(vl_iou)

        if vl_dice > best_dice:
            best_dice = vl_dice
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}/{EPOCHS} | tr_loss {tr_loss:.4f} tr_dice {tr_dice:.4f} | vl_loss {vl_loss:.4f} vl_dice {vl_dice:.4f} iou {vl_iou:.4f}")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _, test_dice, test_iou = evaluate(model, test_loader, criterion, DEVICE)
    print(f"test dice {test_dice:.4f} | test iou {test_iou:.4f}")

    save_plots(history, test_dice, test_iou, exp_name, p_dir)
    save_predictions(model, test_loader, DEVICE, pred_dir, n=8)
    all_results[exp_name] = {"dice": test_dice, "iou": test_iou}

print("\n" + "=" * 55)
print(f"{'experiment':<35} {'dice':>8} {'iou':>8}")
print("=" * 55)
for name, res in all_results.items():
    print(f"{name:<35} {res['dice']:>8.4f} {res['iou']:>8.4f}")
print("=" * 55)

names     = list(all_results.keys())
dice_vals = [all_results[n]["dice"] for n in names]
iou_vals  = [all_results[n]["iou"] for n in names]
x = np.arange(len(names))
w = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w/2, dice_vals, w, label="dice", alpha=0.8)
ax.bar(x + w/2, iou_vals, w, label="iou", alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set_ylabel("score")
ax.legend()
ax.set_title("all experiments test performance")
ax.grid(alpha=0.3, axis="y")
for i, v in enumerate(dice_vals):
    ax.text(i - w/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(EXP_BASE, "summary.png"), dpi=150)
print("summary saved")