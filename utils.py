import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        bce = self.bce(preds, targets)
        p = torch.sigmoid(preds)
        inter = (p * targets).sum(dim=(2, 3))
        dice = 1 - (2 * inter + self.smooth) / (p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)
        return bce + dice.mean()


def dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    inter = (preds * targets).sum(dim=(2, 3))
    return ((2 * inter + smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)).mean().item()


def iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(preds) > threshold).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice = 0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += dice_score(preds.detach(), masks)
    return total_loss / len(loader), total_dice / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            total_loss += criterion(preds, masks).item()
            total_dice += dice_score(preds, masks)
            total_iou += iou_score(preds, masks)
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def save_plots(history, test_dice, test_iou, exp_name, plots_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title(f"{exp_name} loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_dice"], label="train dice")
    plt.plot(epochs, history["val_dice"], label="val dice")
    plt.axhline(y=test_dice, color="red", linestyle="--", label=f"test dice {test_dice:.4f}")
    plt.title(f"{exp_name} dice")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "dice.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["val_iou"], label="val iou", color="green")
    plt.axhline(y=test_iou, color="red", linestyle="--", label=f"test iou {test_iou:.4f}")
    plt.title(f"{exp_name} iou")
    plt.xlabel("epoch")
    plt.ylabel("iou")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "iou.png"), dpi=150)
    plt.close()


def save_predictions(model, loader, device, predictions_dir, n=8):
    model.eval()
    count = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = (torch.sigmoid(model(images)) > 0.5).float()
            for i in range(images.shape[0]):
                if count >= n:
                    return
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
                axes[0].imshow(img, cmap="gray"); axes[0].set_title("input"); axes[0].axis("off")
                axes[1].imshow(masks[i].cpu().squeeze(), cmap="gray"); axes[1].set_title("ground truth"); axes[1].axis("off")
                axes[2].imshow(preds[i].cpu().squeeze(), cmap="gray"); axes[2].set_title("prediction"); axes[2].axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(predictions_dir, f"pred_{count}.png"), dpi=100)
                plt.close()
                count += 1