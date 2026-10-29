"""Part 2: weak supervision with simulated point clicks on PH2.

PH2Clicks is a training dataset that returns, instead of the full mask, only a few
clicked pixels:
  X  [3,H,W]  image (same as part 1)
  Y  [1,H,W]  1 at positive clicks (lesion), 0 everywhere else
  W  [1,H,W]  1 at ALL clicked pixels (positive and negative), 0 elsewhere
The training code passes W where part 1 passed the FOV, so the existing BCE loss
(averaged over pixels where FOV = 1) automatically becomes a loss on the clicked
pixels only (partial cross entropy). Unclicked pixels give no gradient.

The clicks are simulated from the full masks, ONCE when the dataset is created
(fixed seed), like a real annotator who clicks each image once. They are generated on
the resized mask, so their positions match the network input exactly.

Click strategies:
  random - positive clicks uniformly anywhere inside the lesion,
           negative clicks uniformly anywhere in the background
  margin - like a careful annotator: clicks are at least MARGIN pixels away from the
           lesion border (clearly inside / clearly outside), since people rarely click
           exactly on an unclear edge

Check (saves clicks_examples.png):  python3 clicks.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from datasets import PH2


def erode(mask, r):
    """Shrink a binary mask [H,W] by r pixels (min-filter = 1 - max-filter of 1-mask)."""
    m = torch.as_tensor(mask, dtype=torch.float32)[None, None]
    grown_outside = F.max_pool2d(1 - m, kernel_size=2 * r + 1, stride=1, padding=r)
    return (1 - grown_outside)[0, 0].numpy() > 0.5


def sample_points(region, n, rng):
    """Pick n distinct pixels (row, col) from a boolean region."""
    coords = np.argwhere(region)
    idx = rng.choice(len(coords), size=n, replace=len(coords) < n)
    return coords[idx]


def make_clicks(mask, n_pos, n_neg, strategy, rng, margin=10):
    """mask: bool [H,W]. Returns (pos_coords, neg_coords)."""
    fg, bg = mask, ~mask
    if strategy == "margin":
        fg_in, bg_in = erode(fg, margin), erode(bg, margin)
        fg = fg_in if fg_in.any() else fg   # very small lesion: fall back to whole lesion
        bg = bg_in if bg_in.any() else bg
    elif strategy != "random":
        raise ValueError(f"unknown strategy {strategy}")
    return sample_points(fg, n_pos, rng), sample_points(bg, n_neg, rng)


class PH2Clicks(Dataset):
    def __init__(self, split="train", n_pos=5, n_neg=5, strategy="random",
                 margin=10, seed=0, augment=False):
        self.base = PH2(split)          # same IDs, paths, image size and transforms as part 1
        self.augment = augment
        self.clicks = []                # (pos_coords, neg_coords) per image
        H, W = self.base.label_transform(Image.new("L", (10, 10))).shape[1:]
        rng = np.random.default_rng(seed)
        self.Y = torch.zeros(len(self.base), 1, H, W, dtype=torch.uint8)
        self.W = torch.zeros(len(self.base), 1, H, W, dtype=torch.uint8)
        for i, path in enumerate(self.base.label_paths):
            mask = self.base.label_transform(Image.open(path).convert("L"))[0].numpy() > 0.5
            pos, neg = make_clicks(mask, n_pos, n_neg, strategy, rng, margin)
            self.clicks.append((pos, neg))
            pr, pc = torch.as_tensor(pos[:, 0]), torch.as_tensor(pos[:, 1])
            nr, nc = torch.as_tensor(neg[:, 0]), torch.as_tensor(neg[:, 1])
            self.Y[i, 0, pr, pc] = 1
            self.W[i, 0, pr, pc] = 1
            self.W[i, 0, nr, nc] = 1

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        X = self.base.image_transform(Image.open(self.base.image_paths[idx]).convert("RGB"))
        Y, W = self.Y[idx].float(), self.W[idx].float()
        if self.augment:                # flip image and click maps together
            if torch.rand(1) < 0.5:
                X, Y, W = X.flip(-1), Y.flip(-1), W.flip(-1)
            if torch.rand(1) < 0.5:
                X, Y, W = X.flip(-2), Y.flip(-2), W.flip(-2)
        return X, Y, W


# ---------------- Check: draw the clicks on a few training images ----------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = PH2("train")
    sets = {s: PH2Clicks("train", 10, 10, s) for s in ["random", "margin"]}
    print("labelled pixels per image:", int(sets["random"].W[0].sum()),
          "of", sets["random"].W[0].numel())

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for row, (s, ds) in enumerate(sets.items()):
        for col in range(4):
            X, Y, _ = full[col]
            pos, neg = ds.clicks[col]
            ax = axes[row, col]
            ax.imshow(X.permute(1, 2, 0))
            ax.contour(Y[0], levels=[0.5], colors="yellow", linewidths=0.8)
            ax.scatter(pos[:, 1], pos[:, 0], c="lime", s=18, edgecolors="black", linewidths=0.5)
            ax.scatter(neg[:, 1], neg[:, 0], c="red", s=18, edgecolors="black", linewidths=0.5)
            ax.set_title(f"{s}: {full.image_paths[col].split('/')[-1]}", fontsize=9)
            ax.axis("off")
    fig.text(0.5, 0.01, "green = positive click (lesion)   red = negative click (background)   "
             "yellow = true lesion border (not used for training)", ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig("clicks_examples.png", dpi=120)
    print("saved clicks_examples.png")
