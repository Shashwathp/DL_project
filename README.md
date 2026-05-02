# Intracranial Hemorrhage Segmentation

Comparing U-Net variants on the PhysioNet CT-ICH dataset with and without ImageNet pretraining.

## Setup

```bash
pip install torch torchvision segmentation-models-pytorch albumentations matplotlib scikit-learn tqdm numpy Pillow pandas
```

## Data

Download the processed PNG version of the CT-ICH dataset and place it as:

```
data/
  image/   # CT slice PNGs
  label/   # corresponding mask PNGs
```

Update the `IMAGE_DIR` and `MASK_DIR` paths in `train.py`, `inference.py`, and `metrics.py` to point to your data location.

## Files

- `dataset.py` — dataset class and albumentations transforms
- `utils.py` — loss function, dice/iou metrics, train/eval loops, plot and prediction saving
- `train.py` — runs all 6 experiments sequentially, saves weights, plots, and predictions
- `inference.py` — runs inference on 3 specific samples across all models side by side
- `metrics.py` — computes and prints full metrics table on the test set

## Experiments

| # | Model | Encoder | Pretrained |
|---|-------|---------|------------|
| 01 | Vanilla U-Net | ResNet18 | No |
| 02 | Attention U-Net | ResNet18 | No |
| 03 | U-Net++ | ResNet18 | No |
| 04 | Vanilla U-Net | ResNet34 | ImageNet |
| 05 | Attention U-Net | ResNet34 | ImageNet |
| 06 | U-Net++ | ResNet34 | ImageNet |

## Usage

```bash
# train all 6 experiments
python train.py

# run inference on 3 sample images
python inference.py

# print full metrics table
python metrics.py
```

## Output structure

```
experiments/
  01_vanilla_unet/
    weights/best.pth
    plots/loss.png, dice.png, iou.png
    predictions/pred_0.png ... pred_7.png
  02_attention_unet/
    ...
  summary.png
inference_results/
  inference_99.png
  inference_174.png
  inference_548.png
```

## Results

| Model | Dice | IoU | Precision | Recall | F1 |
|-------|------|-----|-----------|--------|----|
| Vanilla U-Net (scratch) | 0.6591 | 0.4916 | 0.7262 | 0.6034 | 0.6591 |
| Attention U-Net (scratch) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| U-Net++ (scratch) | 0.5550 | 0.3841 | 0.6867 | 0.4657 | 0.5550 |
| Vanilla U-Net (pretrained) | **0.7170** | **0.5588** | **0.7895** | **0.6566** | **0.7170** |
| Attention U-Net (pretrained) | 0.6888 | 0.5253 | 0.7492 | 0.6375 | 0.6888 |
| U-Net++ (pretrained) | 0.7010 | 0.5397 | 0.7592 | 0.6511 | 0.7010 |

Key finding: ImageNet pretraining is the dominant factor. Every pretrained model outperforms its scratch counterpart. The Attention U-Net trained from scratch collapsed to predicting empty masks despite a healthy loss curve — a known failure mode in heavily imbalanced segmentation with random initialization.
