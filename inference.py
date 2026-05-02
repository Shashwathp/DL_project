import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_DIR  = "/home/shashwath/Pleural_Effusion/seg/data/image"
MASK_DIR   = "/home/shashwath/Pleural_Effusion/seg/data/label"
EXP_BASE   = "/home/shashwath/Pleural_Effusion/seg/experiments"
OUT_DIR    = "/home/shashwath/Pleural_Effusion/seg/inference_results"
IMAGE_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)

sample_ids = ["99.png", "174.png", "548.png"]

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], is_check_shapes=False)


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


experiments = [
    "01_vanilla_unet",
    "02_attention_unet",
    "03_unetplusplus",
    "04_vanilla_unet_pretrained",
    "05_attention_unet_pretrained",
    "06_unetplusplus_pretrained",
]

print("loading models...")
models = {}
for exp in experiments:
    if os.path.exists(os.path.join(EXP_BASE, exp, "weights", "best.pth")):
        models[exp] = load_model(exp)
        print(f"  {exp}")

for fname in sample_ids:
    image_np = np.array(Image.open(os.path.join(IMAGE_DIR, fname)).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    mask_np  = np.array(Image.open(os.path.join(MASK_DIR, fname)).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST))
    mask_np  = (mask_np > 127).astype(np.float32)

    aug = transform(image=image_np, mask=mask_np)
    image_tensor = aug["image"].unsqueeze(0).to(DEVICE)

    n_cols = 2 + len(models)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(image_np); axes[0].set_title("input ct"); axes[0].axis("off")
    axes[1].imshow(mask_np, cmap="gray"); axes[1].set_title("ground truth"); axes[1].axis("off")

    for i, (exp_name, model) in enumerate(models.items()):
        with torch.no_grad():
            pred = (torch.sigmoid(model(image_tensor)) > 0.5).float().squeeze().cpu().numpy()

        mask_t = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0)
        pred_t = torch.tensor(pred).unsqueeze(0).unsqueeze(0)
        inter  = (pred_t * mask_t).sum()
        dice   = (2 * inter + 1e-6) / (pred_t.sum() + mask_t.sum() + 1e-6)

        axes[2 + i].imshow(pred, cmap="gray")
        axes[2 + i].set_title(f"{exp_name[3:].replace('_', ' ')}\ndice {dice:.3f}", fontsize=8)
        axes[2 + i].axis("off")

    plt.suptitle(fname, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"inference_{fname}"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved inference_{fname}")

print(f"\ndone. results in {OUT_DIR}")