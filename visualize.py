"""Figure for the report: predictions of all trained models on a few TEST images.

Rows: 2 PH2 test images, 2 DRIVE test images.
Columns: image | ground truth | EncDec+BCE | U-Net+BCE | U-Net+focal | U-Net+wBCE
Prediction panels are colour-coded:
  white = correct foreground (TP), red = false alarm (FP), blue = missed (FN),
  black = correct background, grey = outside the FOV (ignored)
Each prediction title shows that image's Dice.

Run on the cluster (CPU is fine):  python3 visualize.py   -> predictions.png
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import PH2, DRIVE
from models import get_model

RUNS = [("encdec", "bce", "EncDec + BCE"),
        ("unet", "bce", "U-Net + BCE"),
        ("unet", "focal", "U-Net + focal"),
        ("unet", "wbce", "U-Net + wBCE")]
N_PER_DATASET = 2


def load(dataset, model, loss):
    m = get_model(model)
    m.load_state_dict(torch.load(f"results/{dataset}_{model}_{loss}.pt", map_location="cpu"))
    return m.eval()


def colour_map(pred, gt, fov):
    img = np.zeros(gt.shape + (3,))
    img[(pred == 1) & (gt == 1)] = [1, 1, 1]          # TP white
    img[(pred == 1) & (gt == 0)] = [0.9, 0.15, 0.15]  # FP red
    img[(pred == 0) & (gt == 1)] = [0.2, 0.45, 1.0]   # FN blue
    img[fov == 0] = [0.35, 0.35, 0.35]                # outside FOV grey
    return img


def dice(pred, gt, fov):
    p, g = pred[fov == 1], gt[fov == 1]
    return 2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8)


rows = []
for name, cls in [("PH2", PH2), ("DRIVE", DRIVE)]:
    ds = cls("test")
    models = [load(name, m, l) for m, l, _ in RUNS]
    for i in range(N_PER_DATASET):
        X, Y, FOV = ds[i]
        gt, fov = Y[0].numpy(), FOV[0].numpy()
        preds = []
        with torch.no_grad():
            for m in models:
                p = (torch.sigmoid(m(X[None]))[0, 0] > 0.5).float().numpy()
                preds.append(p)
        rows.append((name, ds.image_paths[i].split("/")[-1], X, gt, fov, preds))

ncol = 2 + len(RUNS)
fig, axes = plt.subplots(len(rows), ncol, figsize=(2.6 * ncol, 2.4 * len(rows)))
for r, (name, fname, X, gt, fov, preds) in enumerate(rows):
    ax = axes[r]
    ax[0].imshow(X.permute(1, 2, 0).numpy())
    ax[0].set_title(f"{name}: {fname}", fontsize=8)
    ax[1].imshow(colour_map(gt, gt, fov))
    ax[1].set_title("ground truth", fontsize=8)
    for c, (p, (_, _, label)) in enumerate(zip(preds, RUNS)):
        ax[2 + c].imshow(colour_map(p, gt, fov))
        ax[2 + c].set_title(f"{label}\nDice {dice(p, gt, fov):.3f}", fontsize=8)
    for a in ax:
        a.axis("off")

fig.text(0.5, 0.005, "white = correct vessel/lesion   red = false alarm   blue = missed   "
         "black = correct background   grey = outside FOV", ha="center", fontsize=9)
plt.tight_layout(rect=(0, 0.025, 1, 1))
plt.savefig("predictions.png", dpi=150)
print("saved predictions.png")
